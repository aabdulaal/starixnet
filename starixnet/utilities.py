#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, Union

from scipy.signal import find_peaks, periodogram
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd
import warnings


def extract_frequencies(
    ts: Union[pd.Series, pd.DataFrame],
    distance: int = 60,
    min_periods: int = 3,
    max_freqs: int = 10
) -> List[float]:
    """
    Extract frequencies (i.e. periods) from time series using Power Spectral Density (PSD) analysis.

    Args:
        ts (Union[pd.Series, pd.DataFrame]): The time series to inspect for resonant frequencies. Can be multivariate.
        distance (int, optional): Minimal horizontal distance (>= 1) in samples between neighbouring peaks. Refer to
            `scipy.signal.find_peaks` documentation for more information. Defaults to 60.
        min_periods (int, optional): Minimum number of oscillations required for a frequency to be selected. Defaults to
            3.
        max_freqs (int, optional): Maximum dominant periods (i.e. frequencies) to select. Defaults to 10.

    Returns:
        List[float]: A list of the identified peak frequencies.

    """
    if isinstance(ts, pd.DataFrame):
        z = ts.mean(axis=1)
    else:
        z = ts.copy()

    freqs, psd = periodogram(z, fs=1.0, window='hann', scaling='density')
    peak_idxs = find_peaks(psd, distance=distance, height=(0, np.inf))[0]
    peak_order = psd[peak_idxs].argsort()[-min_periods-max_freqs:][::-1]
    peak_idxs = peak_idxs[peak_order]
    if peak_idxs[0] < min_periods:
        print(
            'No common oscillations found. '
            'Either the series does not exhibit periodic patterns or consider reducing min_periods.'
        )
        freqs = []
    elif max(peak_idxs[:min_periods]) >= min_periods:
        idxs = peak_idxs[peak_idxs >= min_periods]
        peak_freqs = [freqs[idx] for idx in idxs[:min(len(idxs), max_freqs)]]
        freqs = [2 * np.pi * f for f in peak_freqs]
        print(f'Found common oscillations at period(s) = {[1 / f for f in peak_freqs]}')
        freqs = [freqs[i] for i in range(max_freqs) if 1 / peak_freqs[i] > distance]  # Only periods > distance
    else:
        print(
            f'Common oscillations found do not satisfy the min_periods={min_periods} requirement. '
            'Consider reducing the min_periods parameter or increase the length of the time series.'
        )
        freqs = []
    return freqs


def minmaxscaler(
    df: pd.DataFrame, scalers: Tuple[pd.DataFrame], scaling_range: Tuple[float, float] = (0.0, 1.0)
) -> pd.DataFrame:
    """
    Apply minmax normalization

    Args:
        df (pd.DataFrame): Raw dataframe.
        scalers (Tuple[pd.DataFrame]): Scalers (min and max) values of the dataframe columns.
        scaling_range (Tuple[float, float], optional): Desired range of scaled data. Defaults to (0.0, 1.0)

    Returns:
        (pd.DataFrame): scaled (normalized) dataframe.

    """
    return (
        (df - scalers[0]) / (scalers[1] - scalers[0])
    ).replace(np.inf, np.nan).fillna(0.0) * (scaling_range[1] - scaling_range[0]) + scaling_range[0]


def tscheck(
    ts: pd.DataFrame,
    min_range_tol: float = 0.05,
    period: Optional[int] = None,
    min_med_tol: float = 5.0,
    drop: bool = False,
) -> List[Union[str, Tuple[str]]]:
    """
    Time series signal quality checks for training SARIXNet.

    Args:
        ts (pd.DataFrame): Time series to inspect for quality issues.
        min_range_tol (float, optional): Minimum tolerance (in percentage) for inspecting the spread (i.e. range) of a
            time series signal. Defaults to 0.05.
        period (Optional[int], optional): Maximumm seasonality in a time series for computing the rolling median. If
            not provided, the time series is assumed nonstationary and rolling distribution checks will be skipped.
            Defaults to None.
        min_med_tol (float, optional): Minimum tolerance for inspecting distribution inconsistencies over time for a
            time series. Defaults to 5.0.
        drop (bool, optional): Whether to drop signals failing the quality checks. Defaults to False.

    Raises:
        ValueError: If time series is too static or impacted by significant data drift.

    Returns:
        List[Union[str, Tuple[str]]]: List of time series features failing the checks.

    """
    dropped_features = []
    for f in ts.columns:
        f_min, f_max = ts[f].min(), ts[f].max()

        # check for static signals
        if not f_max > f_min:
            if drop:
                warnings.warn(
                    f"{f} signal appears static, it will be dropped from the dataset. "
                    "For future runs, consider removing it beforehand"
                )
                dropped_features.append(f)
                continue
            else:
                raise ValueError(f"{f} signal appears static, consider dropping it from the dataset and retry")

        # check for minimum deviation
        min_nonzero = (ts[f][ts[f].abs() > 0.0]).min()
        if abs((f_max - min_nonzero) / min_nonzero) < min_range_tol:
            if drop:
                warnings.warn(
                    f"The non-zero values range of {f} signal is less than the allowed tolerance, "
                    f"{f} will be dropped from the dataset. For future runs, consider removing it beforehand"
                )
                dropped_features.append(f)
                continue
            else:
                raise ValueError(
                    f"The non-zero values range of {f} signal is less than the allowed tolerance, "
                    f"consider dropping {f} from the dataset and retry"
                )

        # check for rolling median (faster than calculating MAD)
        if period:
            rol_med = ts[f].rolling(period).median()
            rol_med_q75, rol_med_q25 = rol_med.quantile(0.75), rol_med.quantile(0.25)
            if rol_med_q25 == 0.0:
                rol_med_q25 = rol_med[rol_med > 0.0].min()
            if rol_med_q75 / rol_med_q25 > min_med_tol:
                if drop:
                    warnings.warn(
                        f"The distribution of {f} signal appears to have shifted over its seasonal period more than the"
                        f" allowed tolerance, {f} will be dropped from the dataset. For future, consider either "
                        f"inspecting the signal for data mining errors or removing it beforehand"
                    )
                    dropped_features.append(f)
                    continue
                else:
                    raise ValueError(
                        f"The distribution of {f} signal appears to have shifted over its seasonal period more than the"
                        f" allowed tolerance, consider correcting or removing {f} from the dataset and retry"
                    )
    assert len(dropped_features) < len(ts.columns), "all signals have been dropped due to failing quality checks"
    return dropped_features


def ts2xyseq(
    ts: pd.DataFrame,
    lagged_steps: int,
    forecast_steps: int,
    lagged_features: List[str],
    forecast_features: Optional[List[str]] = None,
    skip_current_step: bool = True,
    return_x_only: bool = False,
    ts_weights: Optional[pd.DataFrame] = None,
) -> Union[Tuple[pd.DataFrame], pd.DataFrame]:
    """
    Convert time series into a seq-to-seq (i.e. regressors:targets) format for ML training (i.e. x, and y).

    Args:
        ts (pd.DataFrame): Time series to convert into sequence format.
        lagged_steps (int): Number of lagged steps (i.e. regressors) in the seq-to-seq format.
        forecast_steps (int): Number of future steps (i.e. targets) in the seq-to-seq format.
        lagged_features (List[str]): Number of signals (i.e. features) in the input time series.
        forecast_features (Optional[List[str]], optional): Number of signals (i.e. features) in the output time series.
         If not provided, the number of input and output signals is assumed constant. Defaults to None.
        skip_current_step (bool, optional): Whether to skip step (t=0) when converting the time series to a
            seq-to-seq format. For example, assuming `lagged_steps=2` and `forecast_steps=2`, if True, then the
            regressor:target becomes t-2,t-1:t+1,t+2, otherwise it becomes t-2,t-1:t,t+1.. Defaults to True.
        return_x_only (bool, optional): Whether to convert only for regressors (i.e. input sequence). Defaults to False.
        ts_weights (Optional[pd.DataFrame], optional): Univariate time series of sample weights for the target time
            series. If provided, the sample weights would require coversion to the sequence format as well. Defaults to
            None.

    Returns:
        Union[Tuple[pd.DataFrame], pd.DataFrame]: Tuple containing the input sequences (i.e. x), output sequences (i.e.
        y), and the sample weight sequencies (if provided). If `return_x_only=True`, only the input sequences are
        returned.
    """
    # TODO: Consider using multiprocessing for larger datasets, although generators are pretty fast as well
    skip_t = int(skip_current_step)
    if not forecast_features:
        forecast_features = lagged_features
    lagged_ts = pd.concat({'t': ts[lagged_features]}, axis=1)
    lagged_ts_gen = (lagged_ts.shift(i).rename({'t': f't-{i}'}, axis=1) for i in range(1, lagged_steps + 1))
    if return_x_only:
        return pd.concat(
            list(lagged_ts_gen), axis=1
        )[[f't-{i}' for i in reversed(range(1, lagged_steps + 1))]].iloc[lagged_steps:]
    else:
        forecast_ts = pd.concat({'t': ts[forecast_features]}, axis=1)
        forecast_ts_gen = (
            forecast_ts.shift(-i).rename({'t': f't+{i}'}, axis=1) for i in range(skip_t, forecast_steps + skip_t)
        )
        if ts_weights is not None:
            forecast_ts_weights = pd.concat({'t': ts_weights}, axis=1)
            forecast_ts_weights_gen = (
                forecast_ts_weights.shift(-i).rename({'t': f't+{i}'}, axis=1) for i in range(
                    skip_t, forecast_steps + skip_t
                )
            )
            forecast_ts_weights = pd.concat(
                list(forecast_ts_weights_gen), axis=1
            )[[f't+{i}' for i in range(skip_t, forecast_steps + skip_t)]].iloc[lagged_steps:-forecast_steps]
        else:
            forecast_ts_weights = None
        return (
            pd.concat(
                list(lagged_ts_gen), axis=1
            )[[f't-{i}' for i in reversed(range(1, lagged_steps + 1))]].iloc[lagged_steps:-forecast_steps],
            pd.concat(
                list(forecast_ts_gen), axis=1
            )[[f't+{i}' for i in range(skip_t, forecast_steps + skip_t)]].iloc[lagged_steps:-forecast_steps],
            forecast_ts_weights
        )


class Trainer:
    def __init__(
        self,
        optimizer: Optimizer,
        forward_fn: Callable,
        criteria: List[Callable],
        decoder_labels: List[str],
        epochs: int = 100,
        early_stopping_patience: int = -1,
    ):
        """
        Custimizable training module for SARIXNet model.

        Args:
            optimizer (Optimizer): Optimizer from the pytorch library. Refer to `torch.optim` documentation for options.
            forward_fn (Callable): Forward pass function for computing the loss.
            criteria (List[Callable]): Loss functions for the decoders, in order of the decoders.
            decoder_labels (List[str]): Labels of the output decoders in order.
            epochs (int, optional): Number of training passes. Defaults to 100.
            early_stopping_patience (int, optional):  Number of epochs with no improvement after which training will be
                stopped. If passing through all epochs is desired, then this value should be set to -1. Defaults to -1.
        """
        self.optimizer = optimizer
        self.forward_fn = forward_fn
        self.criteria = criteria
        self.decoder_labels = decoder_labels
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        self._message: Dict[str, str] = {}
        self._val_loss: float = 0.0
        self._best_val_loss: float = np.inf
        self._snooze: int = 0
        self._epoch: int = 0

    def _validation(
        self,
        val_dataset: TensorDataset,
        kwargs: List[Dict[str, Any]],
        pbar: tqdm,
        includes_sample_weights: bool = False
    ) -> None:
        if includes_sample_weights:
            y_val_preds = self.forward_fn(val_dataset[:][:-2], kwargs)
            val_loss = [
                criterion(y_val_pred, val_dataset[:][-2], sample_weights=val_dataset[:][-1])
                for criterion, y_val_pred in zip(self.criteria, y_val_preds)
            ]
        else:
            y_val_preds = self.forward_fn(val_dataset[:][:-1], kwargs)
            val_loss = [
                criterion(y_val_pred, val_dataset[:][-1]) for criterion, y_val_pred in zip(self.criteria, y_val_preds)
            ]
        val_loss = sum(val_loss)  # TODO: this is not generalizable to model outputs that are not lists!
        self._val_loss = val_loss.item()
        self._message.update({'val_loss': f'{self._val_loss:.4f}'})
        self._epoch += 1

    def _early_stopping(
        self,
        val_dataset: TensorDataset,
        kwargs: List[Dict[str, Any]],
        pbar: tqdm,
        includes_sample_weights: bool = False
    ) -> None:
        self._validation(val_dataset, kwargs, pbar, includes_sample_weights=includes_sample_weights)
        if self._val_loss >= self._best_val_loss and self._snooze == self.early_stopping_patience:
            self._message.update({'val_loss': ' - '.join((self._message['val_loss'], 'Early stopping triggered'))})
            self._epoch = self.epochs
        elif self._val_loss >= self._best_val_loss:
            self._snooze += 1
        else:
            self._best_val_loss = self._val_loss
            self._snooze = 0

    def _default(
        self, val_dataset: Optional[TensorDataset],
        kwargs: List[Dict[str, Any]],
        pbar: tqdm,
        includes_sample_weights: bool = False
    ) -> None:
        self._epoch += 1

    def train(
        self,
        train_dataloader: DataLoader,
        kwargs: List[Dict[str, Any]],
        val_dataset: Optional[TensorDataset] = None,
        includes_sample_weights: bool = False,
    ):
        """
        Run training process.

        Args:
            train_dataloader (DataLoader): PyTorch dataloader containing training data.
            kwargs (List[Dict[str, Any]]): Additional arguments to pass to forward functions for each item in
                dataloader.
            val_dataset (Optional[TensorDataset], optional): Validation dataset for computing validation loss.
                Recommended when early stopping criteria is triggered. Defaults to None.
            includes_sample_weights (bool, optional): Whether the last item in the dataloader represents a tensor
                dataset for training sample weighting. Defaults to False.
        """
        # TODO: val_dataset can be slow for large sets, modify this to use val dataloader
        if val_dataset is not None:
            if self.early_stopping_patience > 0:
                loop_closer = self._early_stopping
            else:
                loop_closer = self._validation
        else:
            loop_closer = self._default

        with tqdm(total=self.epochs, desc="Training Progress") as pbar:
            while self._epoch < self.epochs:
                for xy in train_dataloader:
                    # reset gradients
                    self.optimizer.zero_grad()

                    if includes_sample_weights:
                        # forward pass
                        y_preds = self.forward_fn(xy[:-2], kwargs)

                        # compute loss
                        loss = [
                            criterion(y_pred, xy[-2], sample_weights=xy[-1])
                            for criterion, y_pred in zip(self.criteria, y_preds)
                        ]
                    else:
                        # forward pass
                        y_preds = self.forward_fn(xy[:-1], kwargs)

                        # compute loss
                        loss = [criterion(y_pred, xy[-1]) for criterion, y_pred in zip(self.criteria, y_preds)]
                    loss = sum(loss)  # TODO: this gives losses equal weight, can try keeping them separate and backpropagate iteratively

                    # backpropagation
                    loss.backward()
                    self.optimizer.step()

                # update progress
                self._message.update({'loss': f'{loss.item():.4f}'})  # TODO: this is only loss from last batch
                loop_closer(val_dataset, kwargs, pbar, includes_sample_weights=includes_sample_weights)
                pbar.set_postfix(self._message)
                pbar.update(1)
