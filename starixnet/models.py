"""STARIXNet — multi-decoder encoder/decoder for time-series forecasting.

Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory of this source
tree.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable
from io import BytesIO
from typing import Any, Self

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from starixnet.losses import QuantileLoss
from starixnet.utilities import (
    Trainer,
    check_signal_quality,
    extract_frequencies,
    make_seq2seq_windows,
    minmax_scale,
)

logger = logging.getLogger(__name__)

Label = str | tuple[str, ...]
ExogFrames = pd.DataFrame | list[pd.DataFrame]
Scalers = tuple[pd.Series, pd.Series]


def _is_multi_exog(exog: ExogFrames | None) -> bool:
    """True iff ``exog`` is a list of frames (i.e. several exogenous groups)."""
    return isinstance(exog, list)


class BaseModel(nn.Module):
    """Shared preprocessing, metadata, and persistence for time-series models."""

    def __init__(
        self,
        lagged_steps: int = 1,
        forecast_steps: int = 1,
        skip_current_step: bool = True,
        enforced_periods: list[int] | None = None,
        min_range_tol: float = 0.05,
        min_med_tol: float = 5.0,
        minmaxscaler_range: tuple[float, float] = (0.0, 1.0),
        **kwargs: Any,
    ) -> None:
        """Initialize shared state.

        Args:
            lagged_steps: Number of past steps used as regressors (encoder length).
            forecast_steps: Number of future steps to forecast (decoder length).
            skip_current_step: When True the target window starts at ``t+1``; else ``t``.
            enforced_periods: Periods to enforce in addition to those discovered via PSD (used by quality checks and the
                seasonal decoder).
            min_range_tol: Minimum relative range tolerance for quality checks.
            min_med_tol: Maximum rolling-median IQR ratio for drift detection.
            minmaxscaler_range: Output range ``(lo, hi)`` for min-max scaling.
            **kwargs: Forwarded to :class:`torch.nn.Module`.
        """
        super().__init__(**kwargs)

        self.lagged_steps = lagged_steps
        self.forecast_steps = forecast_steps
        self.skip_current_step = skip_current_step
        self.enforced_periods = list(enforced_periods) if enforced_periods else None
        self.min_range_tol = min_range_tol
        self.min_med_tol = min_med_tol
        self.minmaxscaler_range = minmaxscaler_range

        self._endog_labels: list[Label] = []
        self._exog_labels: list[Label] | list[list[Label]] = []
        self._endog_removed_labels: list[Label] = []
        self._exog_removed_labels: list[Label] | list[list[Label]] = []
        self._endog_scalers: Scalers | tuple = ()
        self._exog_scalers: Scalers | list[Scalers] | tuple = ()
        self._time_scaler: int = 1
        self._endog_size: int = 0
        self._exog_size: int | list[int] = 0

    # ----------------------------------------------------------------------------------------------------------- public

    def fit(
        self,
        endog: pd.DataFrame,
        exog: ExogFrames | None = None,
        inspect_data_quality: bool = False,
        drop_bad_features: bool = False,
    ) -> tuple[pd.DataFrame, ExogFrames | None]:
        """Run pre-training inspection and register feature metadata.

        Args:
            endog: Target frame.
            exog: Optional exogenous frame or list of frames.
            inspect_data_quality: When True, run :func:`check_signal_quality` on each frame.
            drop_bad_features: When True, drop columns that fail checks; else raise.

        Returns:
            The (possibly column-pruned) endog/exog tuple.
        """
        if inspect_data_quality:
            endog, exog = self._inspect_data_quality(endog, exog, drop_bad_features=drop_bad_features)
        self._register_feature_metadata(endog, exog=exog)
        return endog, exog

    def save(self, filepath: str | BytesIO, compress: bool = True) -> None:
        """Persist model configuration to ``filepath`` via :mod:`joblib`."""
        dump(self.get_config(), filepath, compress=compress)

    def get_config(self) -> dict[str, Any]:
        """Return a serializable snapshot of all instance state."""
        return {
            "lagged_steps": self.lagged_steps,
            "forecast_steps": self.forecast_steps,
            "skip_current_step": self.skip_current_step,
            "min_range_tol": self.min_range_tol,
            "min_med_tol": self.min_med_tol,
            "minmaxscaler_range": self.minmaxscaler_range,
            "_endog_labels": self._endog_labels,
            "_exog_labels": self._exog_labels,
            "_endog_removed_labels": self._endog_removed_labels,
            "_exog_removed_labels": self._exog_removed_labels,
            "_endog_scalers": self._endog_scalers,
            "_exog_scalers": self._exog_scalers,
            "_time_scaler": self._time_scaler,
            "_endog_size": self._endog_size,
            "_exog_size": self._exog_size,
        }

    # ---------------------------------------------------------------------------------------------------------- private

    def _inspect_data_quality(
        self,
        endog: pd.DataFrame,
        exog: ExogFrames | None,
        drop_bad_features: bool = False,
    ) -> tuple[pd.DataFrame, ExogFrames | None]:
        self._endog_removed_labels = self._check_signal_quality(endog, drop_bad_features=drop_bad_features)
        endog = endog.drop(self._endog_removed_labels, axis=1)

        if exog is None:
            return endog, None

        if _is_multi_exog(exog):
            self._exog_removed_labels = []
            cleaned: list[pd.DataFrame] = []
            for frame in exog:
                bad = self._check_signal_quality(frame, drop_bad_features=drop_bad_features)
                self._exog_removed_labels.append(bad)
                cleaned.append(frame.drop(bad, axis=1))
            return endog, cleaned

        self._exog_removed_labels = self._check_signal_quality(exog, drop_bad_features=drop_bad_features)
        return endog, exog.drop(self._exog_removed_labels, axis=1)

    def _check_signal_quality(self, df: pd.DataFrame, drop_bad_features: bool = False) -> list[Label]:
        return check_signal_quality(
            df,
            min_range_tol=self.min_range_tol,
            period=max(self.enforced_periods) if self.enforced_periods else None,
            min_med_tol=self.min_med_tol,
            drop=drop_bad_features,
        )

    def _register_feature_metadata(
        self,
        endog: pd.DataFrame,
        exog: ExogFrames | None = None,
    ) -> None:
        self._endog_labels = endog.columns.to_list()
        self._endog_size = endog.shape[1]
        if exog is None:
            return
        if _is_multi_exog(exog):
            self._exog_labels = [e.columns.to_list() for e in exog]
            self._exog_size = [e.shape[1] for e in exog]
        else:
            self._exog_labels = exog.columns.to_list()
            self._exog_size = exog.shape[1]

    def _preprocess_fit(
        self,
        endog: pd.DataFrame,
        exog: ExogFrames | None,
    ) -> Self:
        self._endog_scalers = (endog.min(), endog.max())
        if exog is not None:
            if _is_multi_exog(exog):
                self._exog_scalers = [(e.min(), e.max()) for e in exog]
            else:
                self._exog_scalers = (exog.min(), exog.max())

        epochtime = endog.index.astype(int)[:2]
        self._time_scaler = int(epochtime[1] - epochtime[0])
        return self

    def _preprocess_transform(
        self,
        endog: pd.DataFrame,
        exog: ExogFrames | None,
    ) -> tuple[pd.DataFrame, ExogFrames | None]:
        endog = self._scale_frame(endog, self._endog_scalers)
        if exog is None:
            return endog, None
        if _is_multi_exog(exog):
            exog = [self._scale_frame(e, s) for e, s in zip(exog, self._exog_scalers, strict=True)]
        else:
            exog = self._scale_frame(exog, self._exog_scalers)
        return endog, exog

    def _scale_frame(self, df: pd.DataFrame, scalers: Scalers) -> pd.DataFrame:
        scaled = minmax_scale(df, scalers, scaling_range=self.minmaxscaler_range)
        scaled.index = scaled.index.astype(int) // self._time_scaler
        return scaled

    def _inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        """Invert min-max scaling on a stack of predictions (clipped to ``>= 0``)."""
        lo, hi = self.minmaxscaler_range
        scaler_min = self._endog_scalers[0].to_numpy()
        scaler_max = self._endog_scalers[1].to_numpy()
        unscaled = (scaled - lo) / (hi - lo) * (scaler_max - scaler_min) + scaler_min
        return np.clip(unscaled, 0.0, np.inf)


class STARIXNet(BaseModel):
    """Multi-decoder forecaster (Seasonal, Temporal, AutoRegressive, eXogenous).

    The model assembles up to four independent decoder branches at build time; the active set is controlled by
    :attr:`applied_attributes`. Each decoder produces a forecast of shape ``(batch, forecast_steps, endog_size)`` and
    the branches are aggregated at prediction time via a quantile reduction.
    """

    SUPPORTED_ATTRIBUTES: tuple[str, ...] = ("seasonal", "temporal", "autoregressive", "exogenous")

    def __init__(
        self,
        n_temporal_embeddings: int,
        embedding_dim: int | None = None,
        enforced_periods: list[int] | None = None,
        max_periods: int = 3,
        min_peak_distance: int = 60,
        min_oscillations: int = 3,
        n_temporal_hidden_layers: int = 1,
        n_ar_hidden_layers: int = 1,
        n_exog_hidden_layers: int = 1,
        trainable_freqs: bool = False,
        fit_multivariate: bool = False,
        endog_filters: int | None = None,
        endog_filters_growth: int = 1,
        endog_kernel_size: int | None = None,
        exog_filters: int | list[int] | None = None,
        exog_filters_growth: int = 1,
        exog_kernel_size: int | None = None,
        ar_encoder: str = "conv",
        ar_d_model: int = 16,
        ar_nhead: int = 4,
        ar_dim_feedforward: int | None = None,
        applied_attributes: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize STARIXNet.

        Args:
            n_temporal_embeddings: Number of distinct temporal categories (e.g. 7 for daily series with weekly
                seasonality).
            embedding_dim: Embedding vector size. Defaults to ``min(50, (n_temporal_embeddings + 1) // 2)`` — the
                pytorch-forecasting heuristic.
            enforced_periods: Periods (sample counts) to enforce in addition to those discovered via PSD analysis.
            max_periods: Maximum number of dominant periods to extract from PSD.
            min_peak_distance: Minimum sample distance between PSD peaks.
            min_oscillations: Minimum oscillation count for a period to qualify.
            n_temporal_hidden_layers: Depth of the temporal MLP.
            n_ar_hidden_layers: Depth of the autoregressive decoder.
            n_exog_hidden_layers: Depth of the exogenous decoder.
            trainable_freqs: Allow PSD frequencies to be updated by gradient.
            fit_multivariate: Use shared (Conv1d) decoders across targets; if False, each target has its own
                per-target MLP head.
            endog_filters: Output channels for the AR Conv1d (multivariate only). Defaults to ``endog_size``.
            endog_filters_growth: Growth ratio across AR hidden layers.
            endog_kernel_size: Kernel size for the AR Conv1d. Defaults to ``min(50, (lagged_steps + 1) // 2)``.
            exog_filters: Output channels for the exogenous Conv1d. Defaults to ``exog_size`` (per group when multiple
                groups are present).
            exog_filters_growth: Growth ratio across exogenous hidden layers.
            exog_kernel_size: Kernel size for the exogenous Conv1d. Defaults to ``min(50, (lagged_steps + 1) // 2)``.
            ar_encoder: Backbone for the autoregressive decoder; one of ``"conv"`` (Conv1d + AdaptivePool — default,
                lightweight, fixed receptive field) or ``"transformer"`` (self-attention over the differenced lagged
                window — heavier per step but able to attend non-uniformly across the window). The exogenous decoder
                always uses Conv1d.
            ar_d_model: Token embedding dimension used when ``ar_encoder="transformer"``. Ignored otherwise.
            ar_nhead: Number of attention heads when ``ar_encoder="transformer"``; must divide ``ar_d_model``.
            ar_dim_feedforward: Hidden width of each transformer block's feed-forward sub-layer. Defaults to
                ``2 * ar_d_model`` (kept small to preserve the lightweight envelope).
            applied_attributes: Subset of ``SUPPORTED_ATTRIBUTES`` to build. Defaults to all four.
            **kwargs: Forwarded to :class:`BaseModel`.
        """
        super().__init__(enforced_periods=enforced_periods, **kwargs)

        if applied_attributes is None:
            applied_attributes = list(self.SUPPORTED_ATTRIBUTES)
        unknown = set(applied_attributes) - set(self.SUPPORTED_ATTRIBUTES)
        if unknown:
            raise ValueError(f"unknown attributes: {sorted(unknown)}")

        if ar_encoder not in {"conv", "transformer"}:
            raise ValueError(f"ar_encoder must be 'conv' or 'transformer', got {ar_encoder!r}")
        if ar_encoder == "transformer" and ar_d_model % ar_nhead != 0:
            raise ValueError(f"ar_d_model={ar_d_model} must be divisible by ar_nhead={ar_nhead}")

        self.n_temporal_embeddings = n_temporal_embeddings
        self.embedding_dim = embedding_dim
        self.max_periods = max_periods
        self.min_peak_distance = min_peak_distance
        self.min_oscillations = min_oscillations
        self.n_temporal_hidden_layers = n_temporal_hidden_layers
        self.n_ar_hidden_layers = n_ar_hidden_layers
        self.n_exog_hidden_layers = n_exog_hidden_layers
        self.trainable_freqs = trainable_freqs
        self.fit_multivariate = fit_multivariate
        self.endog_filters = endog_filters
        self.endog_filters_growth = endog_filters_growth
        self.endog_kernel_size = endog_kernel_size
        self.exog_filters = exog_filters
        self.exog_filters_growth = exog_filters_growth
        self.exog_kernel_size = exog_kernel_size
        self.ar_encoder = ar_encoder
        self.ar_d_model = ar_d_model
        self.ar_nhead = ar_nhead
        self.ar_dim_feedforward = ar_dim_feedforward if ar_dim_feedforward is not None else 2 * ar_d_model
        self.applied_attributes = list(applied_attributes)

        self._freqs: list[float] = []
        self._params: nn.ParameterDict = nn.ParameterDict()
        self._moduls: nn.ModuleDict = nn.ModuleDict()
        self._forward_logic: dict[str, Callable[..., torch.Tensor]] = {}
        self._decoder_labels: list[str] = []

        self._flatten = nn.Flatten()
        self._relu = nn.ReLU()

    # ----------------------------------------------------------------------------------------------------------- public

    def fit(
        self,
        endog: pd.DataFrame,
        exog: ExogFrames | None = None,
        loss_q: float | Iterable[float] = 0.5,
        batch_size: int = 32,
        epochs: int = 100,
        shuffle: bool = True,
        early_stopping_patience: int = -1,
        validation_split: float = 0.0,
        sample_weights: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> Self:
        """Train (or continue training) the network by minimizing quantile loss.

        Args:
            endog: Target frame.
            exog: Optional exogenous frame or list of frames.
            loss_q: Quantile target. Scalar applies to all decoders; iterable must have the same length as built
                decoders.
            batch_size: Samples per gradient step.
            epochs: Maximum number of training passes.
            shuffle: Whether to shuffle batches each epoch.
            early_stopping_patience: Validation patience in epochs (``<= 0`` to disable).
            validation_split: Fraction of the tail used for validation.
            sample_weights: Optional per-sample weight series aligned to ``endog``.
            **kwargs: Forwarded to :meth:`BaseModel.fit`.

        Returns:
            ``self`` after training.
        """
        endog, exog = super().fit(endog, exog=exog, **kwargs)
        endog, exog = self._preprocess_fit(endog, exog)._preprocess_transform(endog, exog)

        # _forward_logic is populated by every decoder branch; _params alone only catches "seasonal" being active.
        if not self._forward_logic:
            self._freqs = self._extract_frequencies(endog)
            if self._endog_size == 1 and self.fit_multivariate:
                warnings.warn(
                    "fit_multivariate=True has no effect with a single target; falling back to univariate.",
                    stacklevel=2,
                )
                self.fit_multivariate = False
            self._build()

        # build sequence frames; ``x_time`` mirrors the future window
        x_time = self._to_sequence_windows(
            pd.DataFrame(endog.index),
            self.forecast_steps,
            self.forecast_steps,
            includes_target=False,
        ).iloc[: -self.forecast_steps]
        x_endog, y, y_sample_weights = self._to_sequence_windows(
            endog,
            self.lagged_steps,
            self.forecast_steps,
            sample_weights=sample_weights,
        )
        x_exog = self._build_exog_sequences(exog, trim_tail=True)

        if exog is None and "exogenous" in self.applied_attributes:
            self.applied_attributes = [a for a in self.applied_attributes if a != "exogenous"]

        x_time, x_endog, x_exog, y, y_sample_weights = self._align_window_sizes(
            x_time, x_endog, x_exog, y, y_sample_weights
        )

        train_set, val_set, fwd_kwargs = self._prepare_tensor_datasets(
            x_time,
            x_endog,
            x_exog,
            y,
            y_sample_weights,
            validation_split,
        )
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

        criteria = self._build_criteria(loss_q)
        trainer = Trainer(
            torch.optim.Adam(self.parameters()),
            self._forward,
            criteria,
            decoder_labels=self._decoder_labels,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
        )
        trainer.train(loader, fwd_kwargs, val_set, includes_sample_weights=sample_weights is not None)

        self._assert_no_nan_params()
        return self

    def predict(
        self,
        endog: pd.DataFrame,
        exog: ExogFrames | None = None,
        batch_size: int = 32,
        aggregated: bool = True,
        aggregation_q: float = 0.5,
    ) -> np.ndarray | list[np.ndarray]:
        """Forecast future steps from a window of past data.

        Args:
            endog: Past values of the targets.
            exog: Past values of the exogenous signals.
            batch_size: Unused (kept for API compatibility); the full window is processed in a single forward pass.
            aggregated: When True, return a quantile-aggregated forecast across decoder outputs; else return the list
                of per-decoder forecasts.
            aggregation_q: Quantile used when ``aggregated=True``.

        Returns:
            Either a single ``(samples, forecast_steps, endog_size)`` array or a list of such arrays (one per active
            decoder).
        """
        endog = self._select_endog_columns(endog)
        exog = self._select_exog_columns(exog)

        endog, exog = self._preprocess_transform(endog, exog)

        x_time = self._to_sequence_windows(
            pd.DataFrame(endog.index),
            self.forecast_steps,
            self.forecast_steps,
            includes_target=False,
        )
        x_endog = self._to_sequence_windows(endog, self.lagged_steps, self.forecast_steps, includes_target=False)
        x_exog = self._build_exog_sequences(exog)

        x_time, x_endog, x_exog, _, _ = self._align_window_sizes(x_time, x_endog, x_exog, None, None, predict=True)

        dataset, _, fwd_kwargs = self._prepare_tensor_datasets(x_time, x_endog, x_exog)
        with torch.no_grad():
            y_preds = self._forward(dataset[:], fwd_kwargs)

        y_arrays = [self._inverse_transform(t.detach().cpu().numpy()) for t in y_preds]
        if aggregated:
            return np.quantile(y_arrays, aggregation_q, axis=0)
        return y_arrays

    def get_config(self) -> dict[str, Any]:
        """Return a serializable snapshot extended with STARIXNet-specific state."""
        config = super().get_config()
        config.update(
            {
                "n_temporal_embeddings": self.n_temporal_embeddings,
                "embedding_dim": self.embedding_dim,
                "enforced_periods": self.enforced_periods,
                "max_periods": self.max_periods,
                "min_peak_distance": self.min_peak_distance,
                "min_oscillations": self.min_oscillations,
                "n_temporal_hidden_layers": self.n_temporal_hidden_layers,
                "n_ar_hidden_layers": self.n_ar_hidden_layers,
                "n_exog_hidden_layers": self.n_exog_hidden_layers,
                "trainable_freqs": self.trainable_freqs,
                "fit_multivariate": self.fit_multivariate,
                "endog_filters": self.endog_filters,
                "endog_filters_growth": self.endog_filters_growth,
                "endog_kernel_size": self.endog_kernel_size,
                "exog_filters": self.exog_filters,
                "exog_filters_growth": self.exog_filters_growth,
                "exog_kernel_size": self.exog_kernel_size,
                "ar_encoder": self.ar_encoder,
                "ar_d_model": self.ar_d_model,
                "ar_nhead": self.ar_nhead,
                "ar_dim_feedforward": self.ar_dim_feedforward,
                "applied_attributes": self.applied_attributes,
                "_freqs": self._freqs,
                "_params": self._params,
                "_moduls": self._moduls,
                "_forward_logic": self._forward_logic,
                "_decoder_labels": self._decoder_labels,
            }
        )
        return config

    @classmethod
    def load_model(cls, filepath: str | BytesIO) -> Self:
        """Reconstruct a STARIXNet from the file written by :meth:`save`."""
        config = load(filepath)
        instance = cls(0)
        for param, val in config.items():
            setattr(instance, param, val)
        return instance

    # -------------------------------------------------------------------------------------------------------- build API

    def _build(self) -> Self:
        if "seasonal" in self.applied_attributes:
            self._build_seasonality_decoder()
        if "temporal" in self.applied_attributes:
            self._build_temporal_decoder()
        if "autoregressive" in self.applied_attributes:
            self._build_autoregression_decoder()
        if self._exog_size and "exogenous" in self.applied_attributes:
            self._build_exogenous_branches()
        return self

    def _build_exogenous_branches(self) -> None:
        if not self.exog_kernel_size:
            self.exog_kernel_size = min(50, (self.lagged_steps + 1) // 2)
        increment_kernel = (self.forecast_steps - self.lagged_steps) // (self.n_exog_hidden_layers + 1)

        if isinstance(self._exog_size, list):
            self.exog_filters, increments = self._resolve_multi_exog_filters()
            for i, (size, filters, growth) in enumerate(
                zip(self._exog_size, self.exog_filters, increments, strict=True)
            ):
                self._build_exogenous_decoder(
                    size,
                    filters,
                    increment_kernel,
                    increment_filters=growth,
                    suffix=f"_{i + 1}",
                )
        else:
            self.exog_filters = self._exog_size if self._exog_size > 1 else None
            self._build_exogenous_decoder(self._exog_size, self.exog_filters, increment_kernel)

    def _resolve_multi_exog_filters(self) -> tuple[list[int | None], list[int]]:
        sizes: list[int] = list(self._exog_size)  # type: ignore[arg-type]
        if self.exog_filters is None:
            if self.fit_multivariate:
                increments = [(self._endog_size - s) // (self.n_exog_hidden_layers + 1) for s in sizes]
                filters: list[int | None] = [s + inc for s, inc in zip(sizes, increments, strict=True)]
            else:
                increments = [0] * len(sizes)
                filters = [s if s > 1 else None for s in sizes]
        else:
            increments = [0] * len(sizes)
            filters = list(self.exog_filters)  # type: ignore[arg-type]
        return filters, increments

    def _build_seasonality_decoder(self) -> None:
        if self._freqs:
            tensor = torch.tensor(self._freqs, dtype=torch.float32)
            self._params["freqs"] = nn.Parameter(tensor, requires_grad=self.trainable_freqs)
            n_freqs = len(self._freqs)
        else:
            n_freqs = self.max_periods
            self._params["freqs"] = nn.Parameter(torch.abs(torch.randn(n_freqs)))
        self._params["amps"] = nn.Parameter(torch.zeros(n_freqs, self._endog_size))
        self._params["phases"] = nn.Parameter(torch.zeros(n_freqs, self._endog_size))
        self._params["disps"] = nn.Parameter(torch.zeros(self._endog_size))
        self._forward_logic["seasonal"] = self._forward_seasonality
        self._decoder_labels.append("seasonal")

    def _build_temporal_decoder(self) -> None:
        if not self.embedding_dim:
            self.embedding_dim = min(50, (self.n_temporal_embeddings + 1) // 2)
        self._moduls["embds"] = nn.Embedding(self.n_temporal_embeddings, self.embedding_dim)
        self._moduls["temporal_hlayers"] = nn.ModuleDict()
        increment = (self.embedding_dim - self._endog_size) // (self.n_temporal_hidden_layers + 1)
        input_dim = self.embedding_dim
        for hl in range(self.n_temporal_hidden_layers):
            output_dim = input_dim - increment
            self._moduls["temporal_hlayers"][f"linear_{hl + 1}"] = nn.Linear(input_dim, output_dim)
            input_dim = output_dim
        self._moduls["temporal_output"] = nn.Linear(input_dim, self._endog_size)
        self._forward_logic["temporal"] = self._forward_temporal
        self._decoder_labels.append("temporal")

    def _build_autoregression_decoder(self) -> None:
        if self.ar_encoder == "transformer":
            self._build_autoregression_transformer_decoder()
        else:
            self._build_autoregression_conv_decoder()
        self._decoder_labels.append("autoregressive")

    def _build_autoregression_conv_decoder(self) -> None:
        self._moduls["ar_hlayers"] = nn.ModuleDict()
        input_dim = self.lagged_steps - 1  # -1 for differencing
        output_dim = self.forecast_steps
        increment = (output_dim - input_dim) // (self.n_ar_hidden_layers + 1)

        if self.fit_multivariate:
            if not self.endog_filters:
                self.endog_filters = self._endog_size
            if not self.endog_kernel_size:
                self.endog_kernel_size = min(50, (input_dim + 1) // 2)
            in_ch, out_ch = self._endog_size, self.endog_filters
            hl = 0
            for hl in range(self.n_ar_hidden_layers):
                output_dim = input_dim + increment
                self._moduls["ar_hlayers"][f"conv_{hl + 1}"] = nn.Conv1d(in_ch, out_ch, self.endog_kernel_size)
                self._moduls["ar_hlayers"][f"pool_{hl + 1}"] = nn.AdaptiveAvgPool1d(output_dim)
                in_ch = out_ch
                out_ch *= self.endog_filters_growth
            self._moduls["ar_hlayers"][f"conv_{hl + 2}"] = nn.Conv1d(in_ch, self._endog_size, self.endog_kernel_size)
            self._moduls["ar_output"] = nn.AdaptiveAvgPool1d(self.forecast_steps)
            self._forward_logic["autoregressive"] = self._forward_autoregression_multivariate
        else:
            for hl in range(self.n_ar_hidden_layers):
                output_dim = input_dim + increment
                self._moduls["ar_hlayers"][f"linear_{hl + 1}"] = nn.ModuleList(
                    nn.Linear(input_dim, output_dim) for _ in range(self._endog_size)
                )
                input_dim = output_dim
            self._moduls["ar_output"] = nn.ModuleList(
                nn.Linear(input_dim, self.forecast_steps) for _ in range(self._endog_size)
            )
            self._forward_logic["autoregressive"] = self._forward_autoregression_univariate

    def _build_autoregression_transformer_decoder(self) -> None:
        # n_ar_hidden_layers is reused as the depth of the transformer stack; nn.TransformerEncoder needs >= 1 block.
        n_layers = max(1, self.n_ar_hidden_layers)
        seq_len = self.lagged_steps - 1  # post-differencing length

        self._moduls["ar_pos_embed"] = nn.Embedding(seq_len, self.ar_d_model)

        if self.fit_multivariate:
            self._moduls["ar_input_proj"] = nn.Linear(self._endog_size, self.ar_d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=self.ar_d_model,
                nhead=self.ar_nhead,
                dim_feedforward=self.ar_dim_feedforward,
                dropout=0.0,
                batch_first=True,
            )
            self._moduls["ar_transformer"] = nn.TransformerEncoder(layer, num_layers=n_layers)
            self._moduls["ar_output_proj"] = nn.Linear(self.ar_d_model, self._endog_size)
            self._moduls["ar_output"] = nn.AdaptiveAvgPool1d(self.forecast_steps)
            self._forward_logic["autoregressive"] = self._forward_autoregression_transformer_multivariate
        else:
            self._moduls["ar_input_projs"] = nn.ModuleList(
                nn.Linear(1, self.ar_d_model) for _ in range(self._endog_size)
            )
            self._moduls["ar_transformers"] = nn.ModuleList(
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.ar_d_model,
                        nhead=self.ar_nhead,
                        dim_feedforward=self.ar_dim_feedforward,
                        dropout=0.0,
                        batch_first=True,
                    ),
                    num_layers=n_layers,
                )
                for _ in range(self._endog_size)
            )
            self._moduls["ar_output"] = nn.ModuleList(
                nn.Linear(self.ar_d_model, self.forecast_steps) for _ in range(self._endog_size)
            )
            self._forward_logic["autoregressive"] = self._forward_autoregression_transformer_univariate

    def _build_exogenous_decoder(
        self,
        exog_size: int,
        exog_filters: int | None,
        increment_kernel: int,
        increment_filters: int = 0,
        suffix: str = "",
    ) -> None:
        hlayers_key = f"exog{suffix}_hlayers"
        output_key = f"exog{suffix}_output"
        self._moduls[hlayers_key] = nn.ModuleDict()

        if self.fit_multivariate:
            in_ch, out_ch = exog_size, exog_filters
            hl = 0
            for hl in range(self.n_exog_hidden_layers):
                output_dim = self.lagged_steps + increment_kernel
                self._moduls[hlayers_key][f"conv_{hl + 1}"] = nn.Conv1d(in_ch, out_ch, self.exog_kernel_size)
                self._moduls[hlayers_key][f"pool_{hl + 1}"] = nn.AdaptiveMaxPool1d(output_dim)
                in_ch = out_ch
                out_ch = (out_ch + increment_filters) * self.exog_filters_growth
            self._moduls[hlayers_key][f"conv_{hl + 2}"] = nn.Conv1d(in_ch, self._endog_size, self.exog_kernel_size)
            self._moduls[output_key] = nn.AdaptiveMaxPool1d(self.forecast_steps)
            self._forward_logic[f"exogenous{suffix}"] = self._forward_exogenous_multivariate
        else:
            if exog_size > 1:
                self._moduls[f"exog{suffix}_conv"] = nn.Conv1d(exog_size, exog_filters, self.exog_kernel_size)
                self._moduls[f"exog{suffix}_pool"] = nn.MaxPool1d(max(1, self.exog_kernel_size // self.forecast_steps))
                input_dim = self._compute_univariate_exog_dim(suffix, exog_filters)
            else:
                input_dim = self.lagged_steps * exog_size
            for hl in range(self.n_exog_hidden_layers):
                output_dim = input_dim + increment_kernel
                self._moduls[hlayers_key][f"linear_{hl + 1}"] = nn.ModuleList(
                    nn.Linear(input_dim, output_dim) for _ in range(self._endog_size)
                )
                input_dim = output_dim
            self._moduls[output_key] = nn.ModuleList(
                nn.Linear(input_dim, self.forecast_steps) for _ in range(self._endog_size)
            )
            self._forward_logic[f"exogenous{suffix}"] = self._forward_exogenous_univariate
        self._decoder_labels.append(f"exogenous{suffix}")

    def _compute_univariate_exog_dim(self, suffix: str, exog_filters: int) -> int:
        conv = self._moduls[f"exog{suffix}_conv"]
        pool = self._moduls[f"exog{suffix}_pool"]
        # output of Conv1d with default stride/padding/dilation
        after_conv = self.lagged_steps - (conv.kernel_size[0] - 1)
        # output of MaxPool1d with default stride=kernel_size
        after_pool = ((after_conv - (pool.kernel_size - 1) - 1) // pool.stride) + 1
        return exog_filters * after_pool

    # ---------------------------------------------------------------------------------------------------------- forward

    def _forward(
        self,
        x: list[torch.Tensor],
        kwargs: dict[str, dict[str, Any]],
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        if "seasonal" in self.applied_attributes:
            outputs.append(self._forward_logic["seasonal"](x[0], **kwargs.get("seasonal", {})))
        if "temporal" in self.applied_attributes:
            outputs.append(self._forward_logic["temporal"](x[0], **kwargs.get("temporal", {})))
        if "autoregressive" in self.applied_attributes:
            outputs.append(self._forward_logic["autoregressive"](x[1], **kwargs.get("autoregressive", {})))
        if "exogenous" in self.applied_attributes:
            if isinstance(self._exog_size, list):
                for i in range(len(self._exog_size)):
                    key = f"exogenous_{i + 1}"
                    outputs.append(self._forward_logic[key](x[2 + i], **kwargs.get(key, {})))
            else:
                outputs.append(self._forward_logic["exogenous"](x[2], **kwargs.get("exogenous", {})))
        return outputs

    def _forward_seasonality(self, x_t: torch.Tensor) -> torch.Tensor:
        # x_t: (B, T, 1); freqs: (F,); phases, amps: (F, O); disps: (O,)
        b, t, _ = x_t.shape
        freqs = self._params["freqs"].view(1, 1, -1, 1)  # (1, 1, F, 1)
        phases = self._params["phases"].unsqueeze(0).unsqueeze(0)  # (1, 1, F, O)
        amps = self._params["amps"].unsqueeze(0).unsqueeze(0)  # (1, 1, F, O)
        x = x_t.view(b, t, 1, 1)
        wave = torch.sin(freqs * x + phases)
        return (amps * wave).sum(dim=2) + self._params["disps"]

    def _forward_temporal(self, x_t: torch.Tensor) -> torch.Tensor:
        flat = x_t.long().reshape(-1)  # (B*T,)
        embedded = self._moduls["embds"](flat % self.n_temporal_embeddings)  # (B*T, E)
        h = embedded
        for layer in self._moduls["temporal_hlayers"].values():
            h = self._relu(layer(h))
        out = self._moduls["temporal_output"](h)
        return out.view(-1, self.forecast_steps, self._endog_size)

    def _forward_autoregression_univariate(self, x_a: torch.Tensor) -> torch.Tensor:
        diff = x_a[:, 1:, :] - x_a[:, :-1, :]
        per_target = [self._flatten(s) for s in torch.split(diff, 1, dim=-1)]
        for hl in range(self.n_ar_hidden_layers):
            layers = self._moduls["ar_hlayers"][f"linear_{hl + 1}"]
            per_target = [self._relu(layer(h)) for layer, h in zip(layers, per_target, strict=True)]
        per_target = [head(h).unsqueeze(-1) for head, h in zip(self._moduls["ar_output"], per_target, strict=True)]
        y = torch.cat(per_target, dim=-1)
        y = torch.cat([(x_a[:, -1:, :] + y[:, :1, :]), y[:, 1:, :]], dim=1)
        return torch.cumsum(y, dim=1)

    def _forward_autoregression_multivariate(self, x_a: torch.Tensor) -> torch.Tensor:
        diff = (x_a[:, 1:, :] - x_a[:, :-1, :]).transpose(1, 2)  # (B, C, T)
        h = diff
        last_hl = self.n_ar_hidden_layers - 1
        for hl in range(self.n_ar_hidden_layers):
            h = self._moduls["ar_hlayers"][f"conv_{hl + 1}"](h)
            h = self._moduls["ar_hlayers"][f"pool_{hl + 1}"](h)
        h = self._moduls["ar_hlayers"][f"conv_{last_hl + 2}"](h)
        h = self._moduls["ar_output"](h).transpose(1, 2)  # (B, T, C)
        y = torch.cat([(x_a[:, -1:, :] + h[:, :1, :]), h[:, 1:, :]], dim=1)
        return torch.cumsum(y, dim=1)

    def _forward_autoregression_transformer_univariate(self, x_a: torch.Tensor) -> torch.Tensor:
        diff = x_a[:, 1:, :] - x_a[:, :-1, :]  # (B, T-1, C)
        positions = torch.arange(diff.shape[1], device=diff.device)
        pos = self._moduls["ar_pos_embed"](positions).unsqueeze(0)  # (1, T-1, d_model)
        per_target: list[torch.Tensor] = []
        for c in range(self._endog_size):
            tok = self._moduls["ar_input_projs"][c](diff[:, :, c : c + 1]) + pos
            tok = self._moduls["ar_transformers"][c](tok)
            head = self._moduls["ar_output"][c](tok.mean(dim=1))  # pool over time, then project
            per_target.append(head.unsqueeze(-1))
        y = torch.cat(per_target, dim=-1)
        y = torch.cat([(x_a[:, -1:, :] + y[:, :1, :]), y[:, 1:, :]], dim=1)
        return torch.cumsum(y, dim=1)

    def _forward_autoregression_transformer_multivariate(self, x_a: torch.Tensor) -> torch.Tensor:
        diff = x_a[:, 1:, :] - x_a[:, :-1, :]  # (B, T-1, C)
        positions = torch.arange(diff.shape[1], device=diff.device)
        pos = self._moduls["ar_pos_embed"](positions).unsqueeze(0)  # (1, T-1, d_model)
        h = self._moduls["ar_input_proj"](diff) + pos  # (B, T-1, d_model)
        h = self._moduls["ar_transformer"](h)
        h = self._moduls["ar_output_proj"](h).transpose(1, 2)  # (B, C, T-1)
        h = self._moduls["ar_output"](h).transpose(1, 2)  # (B, forecast_steps, C)
        y = torch.cat([(x_a[:, -1:, :] + h[:, :1, :]), h[:, 1:, :]], dim=1)
        return torch.cumsum(y, dim=1)

    def _forward_exogenous_univariate(self, x_e: torch.Tensor, suffix: str = "") -> torch.Tensor:
        if x_e.shape[-1] > 1:
            h = x_e.transpose(1, 2)
            h = self._relu(self._moduls[f"exog{suffix}_conv"](h))
            h = self._moduls[f"exog{suffix}_pool"](h).transpose(1, 2)
            per_target = [self._flatten(h) for _ in range(self._endog_size)]
        else:
            per_target = [self._flatten(x_e) for _ in range(self._endog_size)]
        for hl in range(self.n_exog_hidden_layers):
            layers = self._moduls[f"exog{suffix}_hlayers"][f"linear_{hl + 1}"]
            per_target = [self._relu(layer(h)) for layer, h in zip(layers, per_target, strict=True)]
        per_target = [
            head(h).unsqueeze(-1) for head, h in zip(self._moduls[f"exog{suffix}_output"], per_target, strict=True)
        ]
        return torch.cat(per_target, dim=-1)

    def _forward_exogenous_multivariate(self, x_e: torch.Tensor, suffix: str = "") -> torch.Tensor:
        h = x_e.transpose(1, 2)
        last_hl = self.n_exog_hidden_layers - 1
        for hl in range(self.n_exog_hidden_layers):
            h = self._relu(self._moduls[f"exog{suffix}_hlayers"][f"conv_{hl + 1}"](h))
            h = self._moduls[f"exog{suffix}_hlayers"][f"pool_{hl + 1}"](h)
        h = self._moduls[f"exog{suffix}_hlayers"][f"conv_{last_hl + 2}"](h)
        h = self._moduls[f"exog{suffix}_output"](h)
        return h.transpose(1, 2)

    # -------------------------------------------------------------------------------------------------------- internals

    def _extract_frequencies(self, endog: pd.DataFrame) -> list[float]:
        freqs = extract_frequencies(
            endog,
            distance=self.min_peak_distance,
            min_periods=self.min_oscillations,
            max_freqs=self.max_periods,
        )
        if self.enforced_periods:
            for period in self.enforced_periods:
                ef = 2.0 * np.pi * (1.0 / period)
                if ef not in freqs:
                    logger.info("Enforcing a period of length %d", period)
                    freqs = [ef, *freqs]
        return freqs

    def _to_sequence_windows(
        self,
        df: pd.DataFrame,
        lagged_steps: int,
        forecast_steps: int,
        includes_target: bool = True,
        sample_weights: pd.DataFrame | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        return make_seq2seq_windows(
            df,
            lagged_steps=lagged_steps,
            forecast_steps=forecast_steps,
            lagged_features=df.columns.to_list(),
            skip_current_step=self.skip_current_step if includes_target else False,
            return_x_only=not includes_target,
            ts_weights=sample_weights,
        )

    def _build_exog_sequences(
        self,
        exog: ExogFrames | None,
        trim_tail: bool = False,
    ) -> ExogFrames | None:
        """Build seq-to-seq exogenous frames.

        ``trim_tail`` chops the last ``forecast_steps`` rows so that exog rows
        align with ``y`` during fit; predict skips this so the most recent
        window is used as input.
        """
        if exog is None:
            return None

        def _seq(frame: pd.DataFrame) -> pd.DataFrame:
            seq = self._to_sequence_windows(
                frame,
                self.lagged_steps,
                self.forecast_steps,
                includes_target=False,
            )
            return seq.iloc[: -self.forecast_steps] if trim_tail else seq

        if _is_multi_exog(exog):
            return [_seq(e) for e in exog]
        return _seq(exog)

    def _align_window_sizes(
        self,
        x_time: pd.DataFrame,
        x_endog: pd.DataFrame,
        x_exog: ExogFrames | None,
        y: pd.DataFrame | None,
        y_sample_weights: pd.DataFrame | None,
        predict: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, ExogFrames | None, pd.DataFrame | None, pd.DataFrame | None]:
        if self.forecast_steps < self.lagged_steps:
            x_time = x_time.iloc[self.lagged_steps - self.forecast_steps :]
            return x_time, x_endog, x_exog, y, y_sample_weights

        if self.forecast_steps > self.lagged_steps:
            # the predict path uses the (negated) historical convention; preserve it
            diff = self.lagged_steps - self.forecast_steps if predict else self.forecast_steps - self.lagged_steps
            x_endog = x_endog.iloc[diff:]
            if x_exog is not None:
                x_exog = [e.iloc[diff:] for e in x_exog] if _is_multi_exog(x_exog) else x_exog.iloc[diff:]
            if y is not None:
                y = y.iloc[diff:]
            if y_sample_weights is not None:
                y_sample_weights = y_sample_weights.iloc[diff:]
        return x_time, x_endog, x_exog, y, y_sample_weights

    def _build_criteria(self, loss_q: float | Iterable[float]) -> list[QuantileLoss]:
        if isinstance(loss_q, Iterable) and not isinstance(loss_q, float):
            return [QuantileLoss(q) for q in loss_q]
        return [QuantileLoss(loss_q) for _ in self._decoder_labels]

    def _prepare_tensor_datasets(
        self,
        x_time: pd.DataFrame,
        x_endog: pd.DataFrame,
        x_exog: ExogFrames | None = None,
        y: pd.DataFrame | None = None,
        y_sample_weights: pd.DataFrame | None = None,
        validation_split: float = 0.0,
    ) -> tuple[TensorDataset, TensorDataset | None, dict[str, dict[str, Any]]]:
        if validation_split > 0.0:
            cutoff = round(validation_split * x_endog.shape[0])
            train_slices = (slice(None, -cutoff), slice(-cutoff, None))
        else:
            train_slices = (slice(None), None)

        train_tensors = self._frames_to_tensors(
            x_time,
            x_endog,
            x_exog,
            y,
            y_sample_weights,
            slice_=train_slices[0],
        )
        val_tensors: list[torch.Tensor] | None = None
        if train_slices[1] is not None:
            val_tensors = self._frames_to_tensors(
                x_time,
                x_endog,
                x_exog,
                y,
                y_sample_weights,
                slice_=train_slices[1],
            )

        kwargs: dict[str, dict[str, Any]] = {"seasonal": {}, "temporal": {}, "autoregressive": {}}
        if x_exog is not None:
            if _is_multi_exog(x_exog):
                for i in range(len(x_exog)):
                    kwargs[f"exogenous_{i + 1}"] = {"suffix": f"_{i + 1}"}
            else:
                kwargs["exogenous"] = {}

        train_set = TensorDataset(*train_tensors)
        val_set = TensorDataset(*val_tensors) if val_tensors is not None else None
        return train_set, val_set, kwargs

    def _frames_to_tensors(
        self,
        x_time: pd.DataFrame,
        x_endog: pd.DataFrame,
        x_exog: ExogFrames | None,
        y: pd.DataFrame | None,
        y_sample_weights: pd.DataFrame | None,
        slice_: slice,
    ) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = [
            self._frame_to_tensor(x_time.iloc[slice_], self.forecast_steps, 1),
            self._frame_to_tensor(x_endog.iloc[slice_], self.lagged_steps, self._endog_size),
        ]
        if x_exog is not None:
            if _is_multi_exog(x_exog):
                for i, e in enumerate(x_exog):
                    tensors.append(self._frame_to_tensor(e.iloc[slice_], self.lagged_steps, self._exog_size[i]))
            else:
                tensors.append(self._frame_to_tensor(x_exog.iloc[slice_], self.lagged_steps, self._exog_size))
        if y is not None:
            tensors.append(self._frame_to_tensor(y.iloc[slice_], self.forecast_steps, self._endog_size))
        if y_sample_weights is not None:
            tensors.append(self._frame_to_tensor(y_sample_weights.iloc[slice_], self.forecast_steps, 1))
        return tensors

    @staticmethod
    def _frame_to_tensor(df: pd.DataFrame, steps: int, features: int) -> torch.Tensor:
        return torch.from_numpy(df.to_numpy(dtype=np.float32)).reshape(-1, steps, features)

    def _select_endog_columns(self, endog: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self._endog_labels if f not in endog.columns]
        if missing:
            raise ValueError(f"endog is missing required columns: {missing}")
        return endog[self._endog_labels]

    def _select_exog_columns(self, exog: ExogFrames | None) -> ExogFrames | None:
        if exog is None:
            return None
        if _is_multi_exog(exog):
            selected: list[pd.DataFrame] = []
            for i, frame in enumerate(exog):
                missing = [f for f in self._exog_labels[i] if f not in frame.columns]
                if missing:
                    raise ValueError(f"exog group {i} is missing required columns: {missing}")
                selected.append(frame[self._exog_labels[i]])
            return selected
        missing = [f for f in self._exog_labels if f not in exog.columns]
        if missing:
            raise ValueError(f"exog is missing required columns: {missing}")
        return exog[self._exog_labels]

    def _assert_no_nan_params(self) -> None:
        for param in self.parameters():
            if torch.isnan(param).any():
                raise RuntimeError(
                    "Training corrupted: NaN found in network parameters. Inspect inputs and configuration."
                )
