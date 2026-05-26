"""Utility helpers for STARIXNet (data prep, quality checks, training loop).

Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory of this source
tree.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, periodogram
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

Tensors = list[torch.Tensor]
ForwardFn = Callable[[Tensors, dict[str, dict[str, Any]]], list[torch.Tensor]]


def extract_frequencies(
    ts: pd.Series | pd.DataFrame,
    distance: int = 60,
    min_periods: int = 3,
    max_freqs: int = 10,
) -> list[float]:
    """Identify dominant angular frequencies via Power Spectral Density analysis.

    The series is reduced to a single channel (column-wise mean for multivariate input) before the periodogram is
    computed. Peaks are ranked by spectral power and filtered such that:

    1. The peak frequency index is at least ``min_periods`` (i.e. the period fits in the series at least
       ``min_periods`` times).
    2. The implied period exceeds ``distance``.

    Args:
        ts: Time series to inspect. Multivariate input is averaged column-wise.
        distance: Minimum horizontal distance (in samples) between neighbouring peaks; passed to
            :func:`scipy.signal.find_peaks`.
        min_periods: Minimum number of oscillations required for a frequency to qualify.
        max_freqs: Maximum number of dominant frequencies to return.

    Returns:
        Angular frequencies (``2π·f``) sorted by descending spectral power. May
        be empty if no qualifying peaks are found.
    """
    series = ts.mean(axis=1) if isinstance(ts, pd.DataFrame) else ts.copy()

    freqs, psd = periodogram(series, fs=1.0, window="hann", scaling="density")
    peak_idxs, _ = find_peaks(psd, distance=distance, height=(0, np.inf))
    if peak_idxs.size == 0:
        logger.info("No peaks identified in PSD; no seasonal frequencies extracted.")
        return []

    # rank peaks by descending power
    ranked = peak_idxs[np.argsort(psd[peak_idxs])[::-1]]

    # require at least `min_periods` oscillations (i.e. peak idx >= min_periods)
    qualifying = ranked[ranked >= min_periods]
    if qualifying.size == 0:
        logger.info(
            "Common oscillations do not satisfy min_periods=%d. Reduce min_periods or use a longer series.",
            min_periods,
        )
        return []

    qualifying = qualifying[:max_freqs]
    peak_freqs = freqs[qualifying]
    # filter periods that exceed the peak-separation distance
    keep = (1.0 / peak_freqs) > distance
    peak_freqs = peak_freqs[keep]
    if peak_freqs.size == 0:
        logger.info("All qualifying peaks have periods <= distance=%d; returning empty.", distance)
        return []

    logger.info("Found common oscillations at period(s) = %s", (1.0 / peak_freqs).tolist())
    return (2.0 * np.pi * peak_freqs).tolist()


def minmax_scale(
    df: pd.DataFrame,
    scalers: tuple[pd.Series, pd.Series],
    scaling_range: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Vectorized min-max normalization with NaN/inf handling.

    Args:
        df: Raw frame.
        scalers: Tuple of ``(min, max)`` per column, typically ``(df.min(), df.max())`` captured at fit time.
        scaling_range: Target range ``(lo, hi)`` for the normalized output.

    Returns:
        Normalized frame; columns with zero range become 0.0.
    """
    lo, hi = scaling_range
    scaled = (df - scalers[0]) / (scalers[1] - scalers[0])
    return scaled.replace(np.inf, np.nan).fillna(0.0) * (hi - lo) + lo


def check_signal_quality(
    ts: pd.DataFrame,
    min_range_tol: float = 0.05,
    period: int | None = None,
    min_med_tol: float = 5.0,
    drop: bool = False,
) -> list[str | tuple[str, ...]]:
    """Quality checks for time-series signals prior to training.

    Three checks are run per column:

    1. **Static** — flat signals (``min == max``) are rejected.
    2. **Spread** — non-zero range less than ``min_range_tol`` (relative).
    3. **Drift** — when ``period`` is given, an IQR ratio of the rolling median exceeding ``min_med_tol`` indicates
       distribution drift.

    Args:
        ts: Frame to inspect.
        min_range_tol: Minimum relative range tolerance for non-zero values.
        period: Window for rolling-median drift detection. Skipped if ``None``.
        min_med_tol: Maximum allowed Q75/Q25 ratio of the rolling median.
        drop: If True, emit a warning and return failing columns; otherwise raise.

    Returns:
        Columns that failed at least one check (empty when ``drop=False`` and no failure occurred — an exception is
        raised on failure instead).

    Raises:
        ValueError: If ``drop=False`` and any column fails a check.
        AssertionError: If every column is flagged for removal.
    """
    dropped: list[str | tuple[str, ...]] = []

    def _flag(column: str | tuple[str, ...], message: str) -> bool:
        """Return True to continue to next column; False to keep this column."""
        if drop:
            warnings.warn(message + " It will be dropped from the dataset.", stacklevel=2)
            dropped.append(column)
            return True
        raise ValueError(message)

    for col in ts.columns:
        series = ts[col]
        s_min, s_max = series.min(), series.max()

        if not s_max > s_min and _flag(col, f"{col!r} signal appears static."):
            continue

        nonzero = series[series.abs() > 0.0]
        if not nonzero.empty:
            min_nonzero = nonzero.min()
            if abs((s_max - min_nonzero) / min_nonzero) < min_range_tol and _flag(
                col, f"{col!r} non-zero range is below min_range_tol={min_range_tol}."
            ):
                continue

        if period is not None:
            rol_med = series.rolling(period).median()
            q25, q75 = rol_med.quantile(0.25), rol_med.quantile(0.75)
            if q25 == 0.0:
                q25 = rol_med[rol_med > 0.0].min()
            if (
                q25
                and q75 / q25 > min_med_tol
                and _flag(col, f"{col!r} rolling-median drift exceeds min_med_tol={min_med_tol}.")
            ):
                continue

    if dropped and len(dropped) >= len(ts.columns):
        raise AssertionError("all signals failed quality checks; nothing left to train on")
    return dropped


def make_seq2seq_windows(
    ts: pd.DataFrame,
    lagged_steps: int,
    forecast_steps: int,
    lagged_features: list[str],
    forecast_features: list[str] | None = None,
    skip_current_step: bool = True,
    return_x_only: bool = False,
    ts_weights: pd.DataFrame | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Convert a time-series frame into seq-to-seq (lagged ↔ future) format.

    Args:
        ts: Wide-format frame indexed by time.
        lagged_steps: Number of past steps to use as regressors.
        forecast_steps: Number of future steps to predict.
        lagged_features: Columns used as lagged regressors.
        forecast_features: Columns used as forecast targets. Defaults to ``lagged_features``.
        skip_current_step: If True, future starts at ``t+1`` (regressor uses steps ``t-L..t-1``); else starts at ``t``.
        return_x_only: When True, only the regressor frame is returned.
        ts_weights: Optional univariate frame of per-step sample weights aligned to ``ts``. When supplied the
            future-window weights are returned.

    Returns:
        Regressor frame if ``return_x_only`` else a 3-tuple of ``(x, y, weights)`` where ``weights`` is ``None`` when
        ``ts_weights`` is ``None``.
    """
    skip_t = int(skip_current_step)
    forecast_features = forecast_features or lagged_features

    x = _stacked_shifts(
        ts[lagged_features],
        lags=range(1, lagged_steps + 1),
        sign="-",
    )
    x_cols = [f"t-{i}" for i in reversed(range(1, lagged_steps + 1))]
    x = x[x_cols]

    if return_x_only:
        return x.iloc[lagged_steps:]

    y = _stacked_shifts(
        ts[forecast_features],
        lags=range(skip_t, forecast_steps + skip_t),
        sign="+",
    )
    y_cols = [f"t+{i}" for i in range(skip_t, forecast_steps + skip_t)]
    y = y[y_cols]

    w = None
    if ts_weights is not None:
        w = _stacked_shifts(ts_weights, lags=range(skip_t, forecast_steps + skip_t), sign="+")
        w = w[y_cols].iloc[lagged_steps:-forecast_steps]

    return (
        x.iloc[lagged_steps:-forecast_steps],
        y.iloc[lagged_steps:-forecast_steps],
        w,
    )


def _stacked_shifts(
    frame: pd.DataFrame,
    lags: Iterable[int],
    sign: str,
) -> pd.DataFrame:
    """Stack shifted copies of ``frame`` column-wise, naming each block ``t{sign}i``."""
    shift_sign = -1 if sign == "+" else 1
    grouped = pd.concat({"t": frame}, axis=1)
    blocks = (grouped.shift(shift_sign * i).rename({"t": f"t{sign}{i}"}, axis=1) for i in lags)
    return pd.concat(list(blocks), axis=1)


class Trainer:
    """Multi-decoder training loop with optional validation and early stopping.

    The forward function is expected to return a list of decoder outputs, one per element in ``criteria`` /
    ``decoder_labels``. Losses are summed across decoders before backpropagation, and an epoch-running mean is shown
    in the progress bar.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        forward_fn: ForwardFn,
        criteria: list[Callable[..., torch.Tensor]],
        decoder_labels: list[str],
        epochs: int = 100,
        early_stopping_patience: int = -1,
    ) -> None:
        """Initialize the trainer.

        Args:
            optimizer: PyTorch optimizer (typically ``torch.optim.Adam``).
            forward_fn: Callable mapping ``(tensor_list, kwargs)`` to a list of decoder predictions.
            criteria: Loss callables, one per decoder.
            decoder_labels: Names of decoders (for progress reporting).
            epochs: Maximum number of training passes.
            early_stopping_patience: Patience in epochs for val-loss plateaus; ``<= 0`` disables early stopping.
        """
        if len(criteria) != len(decoder_labels):
            raise ValueError(f"criteria/decoder_labels mismatch: {len(criteria)} vs {len(decoder_labels)}")
        self.optimizer = optimizer
        self.forward_fn = forward_fn
        self.criteria = criteria
        self.decoder_labels = decoder_labels
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        self._best_val_loss = float("inf")
        self._stale_epochs = 0

    def train(
        self,
        train_dataloader: DataLoader,
        kwargs: dict[str, dict[str, Any]],
        val_dataset: TensorDataset | None = None,
        includes_sample_weights: bool = False,
    ) -> None:
        """Run the training loop.

        Args:
            train_dataloader: Iterates batched tensors with the same shape contract as ``forward_fn`` expects (plus
                targets and optional weights at the end).
            kwargs: Additional keyword args forwarded to ``forward_fn`` per decoder.
            val_dataset: Optional validation set; required for early stopping.
            includes_sample_weights: When True, the last tensor in each batch is treated as per-sample weights and the
                second-to-last as targets.
        """
        do_validation = val_dataset is not None
        do_early_stop = do_validation and self.early_stopping_patience > 0

        with tqdm(total=self.epochs, desc="Training Progress") as pbar:
            for _ in range(self.epochs):
                running_loss, n_batches = 0.0, 0
                for batch in train_dataloader:
                    loss = self._step(batch, kwargs, includes_sample_weights)
                    running_loss += loss
                    n_batches += 1

                msg: dict[str, str] = {"loss": f"{running_loss / max(n_batches, 1):.4f}"}

                if do_validation:
                    val_loss = self._evaluate(val_dataset, kwargs, includes_sample_weights)
                    msg["val_loss"] = f"{val_loss:.4f}"
                    if do_early_stop and self._should_stop(val_loss):
                        msg["val_loss"] += " - early stopping"
                        pbar.set_postfix(msg)
                        pbar.update(1)
                        break

                pbar.set_postfix(msg)
                pbar.update(1)

    def _step(
        self,
        batch: Tensors,
        kwargs: dict[str, dict[str, Any]],
        includes_sample_weights: bool,
    ) -> float:
        """One optimizer step on a batch; returns the total scalar loss."""
        self.optimizer.zero_grad(set_to_none=True)
        if includes_sample_weights:
            inputs, target, weights = batch[:-2], batch[-2], batch[-1]
            preds = self.forward_fn(inputs, kwargs)
            losses = [
                criterion(pred, target, sample_weights=weights)
                for criterion, pred in zip(self.criteria, preds, strict=True)
            ]
        else:
            inputs, target = batch[:-1], batch[-1]
            preds = self.forward_fn(inputs, kwargs)
            losses = [criterion(pred, target) for criterion, pred in zip(self.criteria, preds, strict=True)]
        total = torch.stack(losses).sum()
        total.backward()
        self.optimizer.step()
        return total.item()

    @torch.no_grad()
    def _evaluate(
        self,
        val_dataset: TensorDataset,
        kwargs: dict[str, dict[str, Any]],
        includes_sample_weights: bool,
    ) -> float:
        """Validation pass over the entire dataset; returns the summed loss."""
        all_tensors = val_dataset[:]
        if includes_sample_weights:
            inputs, target, weights = all_tensors[:-2], all_tensors[-2], all_tensors[-1]
            preds = self.forward_fn(inputs, kwargs)
            losses = [
                criterion(pred, target, sample_weights=weights)
                for criterion, pred in zip(self.criteria, preds, strict=True)
            ]
        else:
            inputs, target = all_tensors[:-1], all_tensors[-1]
            preds = self.forward_fn(inputs, kwargs)
            losses = [criterion(pred, target) for criterion, pred in zip(self.criteria, preds, strict=True)]
        return torch.stack(losses).sum().item()

    def _should_stop(self, val_loss: float) -> bool:
        """Track stale validation epochs and return True when patience is exhausted."""
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._stale_epochs = 0
            return False
        self._stale_epochs += 1
        return self._stale_epochs > self.early_stopping_patience
