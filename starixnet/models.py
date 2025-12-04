#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

from collections.abc import Callable, Iterable
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self  # TODO: if only using py311 or later, Self can be in base typing

from joblib import dump, load
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from starixnet.losses import QLoss
from starixnet.utilities import extract_frequencies, minmaxscaler, Trainer, tscheck, ts2xyseq


class BaseModel(nn.Module):
    def __init__(
        self,
        lagged_steps: int = 1,
        forecast_steps: int = 1,
        skip_current_step: bool = True,
        enforced_periods: Optional[List[int]] = None,
        min_range_tol: float = 0.05,
        min_med_tol: float = 5.0,
        minmaxscaler_range: Tuple[float, float] = (0.0, 1.0),
        **kwargs,
    ):
        """
        Base model for timeseries forecasting from which to inherit from.

        Args:
            lagged_steps (int, optional): Number of past steps used to forecast future steps in seq-to-seq approach.
                Also known as 'context length' or 'encoder length' in similar applications. Defaults to 1.
            forecast_steps (int, optional): Number of future steps forecasted from the `lagged_steps` inputs in a
                seq-to-seq approach. Also Known as 'decoder length' in similar applications. Defaults to 1.
            skip_current_step (bool, optional): Whether to skip step (t=0) when converting the time series to a
                seq-to-seq format. For example, assuming `lagged_steps=2` and `forecast_steps=2`, if True, then the
                regressor:target becomes t-2,t-1:t+1,t+2, otherwise it becomes t-2,t-1:t,t+1. Defaults to True.
            enforced_periods (Optional[List[int]], optional): Explictly defined periods (i.e. frequencies) for data
                quality checks. Defaults to None.
            min_range_tol (float, optional):  Minimum tolerance (in percentage) for inspecting the spread (i.e. range)
                of a time series signal. Defaults to 0.05.
            min_med_tol (float, optional): Minimum tolerance for inspecting distribution inconsistencies over time for a
                time series. Defaults to 25.0.
            minmaxscaler_range (Tuple[float, float], optional): Desired range of scaled data. Defaults to (0.0, 1.0)
            **kwargs: Additional arguments to `torch.nn.Module` base model class.
        """
        super(BaseModel, self).__init__(**kwargs)

        self.lagged_steps = lagged_steps
        self.forecast_steps = forecast_steps
        self.skip_current_step = skip_current_step
        self.enforced_periods = enforced_periods
        self.min_range_tol = min_range_tol
        self.min_med_tol = min_med_tol
        self.minmaxscaler_range = minmaxscaler_range

        self._endog_labels: List[Union[str, Tuple[str]]] = []
        self._exog_labels: List[Union[List[Union[str, Tuple[str]]], Union[str, Tuple[str]]]] = []
        self._endog_removed_labels: List[Union[str, Tuple[str]]] = []
        self._exog_removed_labels: List[Union[List[Union[str, Tuple[str]]], Union[str, Tuple[str]]]] = []
        self._endog_scalers: Tuple[pd.DataFrame] = ()
        self._exog_scalers: Union[List[Tuple[pd.DataFrame]], Tuple[pd.DataFrame]] = ()
        self._time_scaler: int = 1
        self._endog_size: int = 0
        self._exog_size: Union[List[int], int] = 0

    def fit(
        self,
        endog: pd.DataFrame,
        exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        inspect_data_quality: bool = False,
        drop_bad_features: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Perform preliminary preparations for model training.

        Args:
            endog (pd.DataFrame): Target time series (i.e. signals).
            exog (Optional[Union[pd.DataFrame, List[pd.DataFrame]]], optional): eXogenous time series to fit targets. If
                not provided, the network will produce seasonal, temporal, and autoregressive-based forecasts only. If a
                List of dataframes is provided, then separte encoders and decoders will be built for each data frame.
                Defaults to None.
            inspect_data_quality (bool, optional): Whether to scan data for quality problems before training. Defaults
                to False.
            drop_bad_features (bool, optional): Whether to drop features from inputs upon failing quality checks
                Defaults to False.

        Returns:
            (Tuple[pd.DataFrame, Optional[pd.DataFrame]]): Input time series after perofrming optional checks.
        """
        # inspect data
        if inspect_data_quality:
            endog, exog = self._inspect_data_quality(endog, exog, drop_bad_features=drop_bad_features)

        # register features metadata
        self._register_feature_metadata(endog, exog=exog)

        return endog, exog

    def save(self, filepath: Union[str, BytesIO], compress: bool = True,) -> None:
        """
        Save model to file or memory bytes.

        Args:
            filepath (Union[str, BytesIO]): Path for storing model file. Can also be stored in memory bytes.
            compress (bool, optional): Whether to compress and reduce file size. Defaults to True.
        """
        file = self.get_config()
        dump(file, filepath, compress=compress)

    def get_config(self) -> Dict[str, Any]:
        """
        Extract model configuration (i.e. parameters, variables, attributes, and settings).

        Returns:
            Dict[str, Any]: Dictionary of all instance variables and their values.
        """
        config = {
            'lagged_steps': self.lagged_steps,
            'forecast_steps': self.forecast_steps,
            'skip_current_step': self.skip_current_step,
            'min_range_tol': self.min_range_tol,
            'min_med_tol': self.min_med_tol,
            'minmaxscaler_range': self.minmaxscaler_range,
            '_endog_labels': self._endog_labels,
            '_exog_labels': self._exog_labels,
            '_endog_removed_labels': self._endog_removed_labels,
            '_exog_removed_labels': self._exog_removed_labels,
            '_endog_scalers': self._endog_scalers,
            '_exog_scalers': self._exog_scalers,
            '_time_scaler': self._time_scaler,
            '_endog_size': self._endog_size,
            '_exog_size': self._exog_size,
        }
        return config

    def _inspect_data_quality(
        self,
        endog: pd.DataFrame,
        exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        drop_bad_features: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[Union[pd.DataFrame, List[pd.DataFrame]]]]:
        # inspect for endogenous features
        self._endog_removed_labels = self._tscheck(endog, drop_bad_features=drop_bad_features)
        endog = endog.drop(self._endog_removed_labels, axis=1)

        # inspect for exogenous features
        if exog is not None:
            if isinstance(exog, list):
                for i, _exog in enumerate(exog):
                    self._exog_removed_labels.append(self._tscheck(_exog, drop_bad_features=drop_bad_features))
                    exog[i] = exog[i].drop(self._exog_removed_labels[i], axis=1)
            else:
                self._exog_removed_labels = self._tscheck(exog, drop_bad_features=drop_bad_features)
                exog = exog.drop(self._exog_removed_labels, axis=1)

        return endog, exog

    def _tscheck(self, df: pd.DataFrame, drop_bad_features: bool = False,) -> List[Union[str, Tuple[str]]]:
        return tscheck(
            df,
            min_range_tol=self.min_range_tol,
            period=max(self.enforced_periods) if self.enforced_periods else None,
            min_med_tol=self.min_med_tol,
            drop=drop_bad_features,
        )

    def _register_feature_metadata(
        self, endog: pd.DataFrame, exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ):
        self._endog_labels = endog.columns.to_list()
        self._endog_size = endog.shape[1]
        if exog is not None:
            if isinstance(exog, list):
                self._exog_labels = [e.columns.to_list() for e in exog]
                self._exog_size = [e.shape[1] for e in exog]
            else:
                self._exog_labels = exog.columns.to_list()
                self._exog_size = exog.shape[1]

    def _preprocess_fit(
        self, endog: pd.DataFrame, exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]]
    ) -> Self:
        # normalization scalers
        self._endog_scalers = (endog.min(), endog.max())
        if exog is not None:
            if isinstance(exog, list):
                self._exog_scalers = [(e.min(), e.max()) for e in exog]
            else:
                self._exog_scalers = (exog.min(), exog.max())

        # time scaler
        epochtime = endog.index.astype(int)[:2]
        self._time_scaler = epochtime[1] - epochtime[0]

        return self

    def _preprocess_transform(
        self, endog: pd.DataFrame, exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]],
    ) -> Tuple[pd.DataFrame, Optional[Union[pd.DataFrame, List[pd.DataFrame]]]]:
        endog = minmaxscaler(endog, self._endog_scalers, scaling_range=self.minmaxscaler_range)
        endog.index = endog.index.astype(int) // self._time_scaler
        if exog is not None:
            if isinstance(exog, list):
                for i in range(len(exog)):
                    exog[i] = minmaxscaler(exog[i], self._exog_scalers[i], scaling_range=self.minmaxscaler_range)
                    exog[i].index = exog[i].index.astype(int) // self._time_scaler
            else:
                exog = minmaxscaler(exog, self._exog_scalers, scaling_range=self.minmaxscaler_range)
                exog.index = exog.index.astype(int) // self._time_scaler
        return endog, exog


class STARIXNet(BaseModel):
    def __init__(
        self,
        n_temporal_embeddings: int,
        embedding_dim: Optional[int] = None,
        enforced_periods: Optional[List[int]] = None,
        max_periods: int = 3,
        min_peak_distance: int = 60,
        min_oscillations: int = 3,
        n_temporal_hidden_layers: int = 1,
        n_ar_hidden_layers: int = 1,
        n_exog_hidden_layers: int = 1,
        trainable_freqs: bool = False,
        fit_multivariate: bool = False,
        endog_filters: Optional[int] = None,
        endog_filters_growth: int = 1,
        endog_kernel_size: Optional[int] = None,
        exog_filters: Optional[Union[int, List[int]]] = None,
        exog_filters_growth: int = 1,
        exog_kernel_size: Optional[int] = None,
        applied_attributes: List[str] = ["seasonal", "temporal", "autoregressive", "exogenous"],
        **kwargs,
    ):
        """
        Encoder-decoder neural (Net)work architecture with multiple decoder outputs for time series forecasts.

        The decoders achieve independant forecasts of the targets based on the (S)easonal, (T)emporal
        patterns, (A)uto(R)egressive characteristics of the (I)ntegrated targets, and their relationships with
        e(X)ogenous features. The name STARIXnet is derived from the terms: Seasonal, Temporal, AutoRegressive,
        Integrated, eXogenous, and neural Network.

        Use STARIXNet if your objective is to obtain multiple independent forecasts based on the various attributes
        of your target data. STARIXNet will compute different forecast options and applies a user-set aggregation method
        appropriate selection to their use case. For example, you may wish to either apply the maximum of all
        forecasts, or some aggregation of them.

        Args:
            n_temporal_embeddings (int): Number of unique temporal features in the time series. For example, a time
                series of daily frequency and weekly seasonality should have this value set to 7 (i.e. there are 7
                unique temporal features, which are the days of the week).
            embedding_dim (Optional[int], optional): Size of the embedding vector encoding the temporal features. If
                None, the size will be calculate as `min(50, (n_temporal_embeddings + 1) // 2)`
                Defaults to None.
            enforced_periods (Optional[List[int]], optional): Explictly defined periods (i.e. frequencies) for the
                seasonal autoencoder to fit to the target signals, on top of whatever periods SARIXNet may extract via
                Power Spectral Density (PSD) analysis. Defaults to None.
            max_periods (int, optional): Maximum limit on the number of dominant periods (i.e. frequencies) to be
                selected from the PSD analysis of the target signals. Defaults to 3.
            min_peak_distance (int, optional): Minimal horizontal distance (>= 1) in samples between neighbouring peaks,
                used in analyzing the PSD in search for dominant periods. For reference, refer to
                `scipy.signal.find_peaks` documentation. Defaults to 60.
            min_oscillations (int, optional): Minimum number of oscillations to identify a dominant period (frequency)
                during the PSD analysis of target signals. Defaults to 3.
            n_temporal_hidden_layers (int, optional): Depth of the autoencoder for temporal forecasts. Defaults to 1.
            n_ar_hidden_layers (int, optional): Depth of the autoencoder for autoregressive-integrated forecasts.
                Defaults to 1.
            n_exog_hidden_layers (int, optional): Depth of the autoencoder for exogenous-based forecasts. Defaults to 1.
            trainable_freqs (bool, optional): Whether to allow extracted frequencies to be updated during training. If
                True, the explictly defined periods and the periods extracted from PSD analysis may significantly change
                and converge to narrow values if it reduces the training loss (not recommended). Defaults to False.
            fit_multivariate (bool, optional): Applicable for when the target data consist of multiple signals. When set
                to `False`, each signal is forecasted independently (i.e. decoder layers branched to separate outputs
                per signal), while `True` indicates using fully connected layers sith a single output decoder of output
                size equal to the number of signals. Defaults to False.
            endog_filters (Optional[int], optional): Number of convolutional filters to apply to the differenced
                endogenous signals in the first autoregressive layer. Only applicable when `fit_multivariate` is True.
                For reference, refer to `torch.nn.Conv1d` in the PyTorch documentation, where this argument translates
                to `out_channels`. If not provided, the number of filters will be set to the number of prediction
                targets. Defaults to None.
            endog_filters_growth (int, optional): Rate of growing/shrinking of the filters throughout the autoregressive
                hidden layers. Only applicable when `fit_multivariate` is True. Defaults to 1.
            endog_kernel_size (Optional[int], optional): Kernel size for convolutional layer applied to the differenced
                endogenous signal. Only applicable when `fit_multivariate` is True. For reference, refer to
                `torch.nn.Conv1d` in the PyTorch documentation, where this argument translates to `kernel_size`.
                Defaults to None.
            exog_filters (Optional[Union[int, List[int]]], optional): Number of convolutional filters to apply to the
                exogenous regressors in the first exogenous layer. For reference, refer to `torch.nn.Conv1d` in the
                PyTorch documentation, where this argument translates to `out_channels`. If not provided, the number of
                filters will be set to the number of exogenous variables. If multiple sets of exogneous variables are
                modeled and `exog_filters` is provided, then it must be a list of the same size as the number of sets.
                Defaults to None.
            exog_filters_growth (int, optional): Rate of growing/shrinking of the filters throughout the exogenous
                hidden layers. Only applicable when `fit_multivariate` is True. Defaults to 1.
            exog_kernel_size (Optional[int], optional): Kernel size for convolutional layer applied ot exogenous signal.
                or reference, refer to `torch.nn.Conv1d` in the PyTorch documentation, where this argument translates
                to `kernel_size`. Defaults to None.
            applied_attributes (List[str], optional): Attributes of the time series desired for modeling. Defaults to [
                "seasonal", "temporal", "autoregressive", "exogenous"].
            **kwargs: Additional arguments to `BaseModel` base model class.
        """
        super(STARIXNet, self).__init__(enforced_periods=enforced_periods, **kwargs)

        self.n_temporal_embeddings = n_temporal_embeddings
        self.embedding_dim = embedding_dim
        self.enforced_periods = enforced_periods
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
        self.applied_attributes = applied_attributes

        self._freqs: List[float] = []
        self._params: nn.Module = nn.ParameterDict()
        self._moduls: nn.Module = nn.ModuleDict()
        self._forward_logic: Dict[str, Callable] = {}
        self._decoder_labels: List[str] = []

        # general-use nn operations
        self._flatten = nn.Flatten()
        self._relu = nn.ReLU()

    def _build(self) -> Self:
        # seasonality encoder-decoder
        if 'seasonal' in self.applied_attributes:
            self._build_seasonality_autoencoder()

        # temporal encoder-decoder
        if 'temporal' in self.applied_attributes:
            self._build_temporal_autoencoder()

        # autoregression encoder-decoder
        if 'autoregressive' in self.applied_attributes:
            self._build_autoregression_autoencoder()

        # exogenous encoder-decoder
        if self._exog_size and 'exogenous' in self.applied_attributes:
            # prepare common parameters
            if not self.exog_kernel_size:
                self.exog_kernel_size = min(50, (self.lagged_steps + 1) // 2)  # TODO: verify that it is a good default
            increment_kernel = (self.forecast_steps - self.lagged_steps) // (self.n_exog_hidden_layers + 1)

            # prepare parameters by exog group, if multiple, and build
            if isinstance(self._exog_size, list):
                if not self.exog_filters:
                    if self.fit_multivariate:
                        increment_filters = [
                            (self._endog_size - exog_size) // (self.n_exog_hidden_layers + 1)
                            for exog_size in self._exog_size
                        ]
                        self.exog_filters = [
                            exog_size + increment for exog_size, increment in zip(self._exog_size, increment_filters)
                        ]
                    else:
                        increment_filters = [0] * len(self._exog_size)
                        self.exog_filters = [exog_size if exog_size > 1 else None for exog_size in self._exog_size]
                else:
                    increment_filters = [0] * len(self._exog_size)
                for i, (exog_size, exog_filters, increment) in enumerate(
                    zip(self._exog_size, self.exog_filters, increment_filters)
                ):
                    self._build_exogenous_autoencoder(
                        exog_size, exog_filters, increment_kernel, increment_filters=increment, suffix=f"_{i+1}"
                    )
            else:
                self.exog_filters = self._exog_size if self._exog_size > 1 else None
                self._build_exogenous_autoencoder(self._exog_size, self.exog_filters, increment_kernel)

        return self

    def _forward(self, x: Iterable[torch.Tensor], kwargs: List[Dict[str, Any]]) -> List[torch.Tensor]:
        outputs = []
        if "seasonal" in self.applied_attributes:
            outputs.append(self._forward_logic['seasonal'](x[0], **kwargs['seasonal']))
        if 'temporal' in self.applied_attributes:
            outputs.append(self._forward_logic['temporal'](x[0], **kwargs['temporal']))
        if 'autoregressive' in self.applied_attributes:
            outputs += [self._forward_logic['autoregressive'](x[1], **kwargs['autoregressive'])]
        if 'exogenous' in self.applied_attributes:
            if isinstance(self._exog_size, list):
                for i in range(len(self._exog_size)):
                    outputs.append(self._forward_logic[f"exogenous_{i+1}"](x[2+i], **kwargs[f"exogenous_{i + 1}"]))
            else:
                outputs.append(self._forward_logic['exogenous'](x[2], **kwargs['exogenous']))
        return outputs
         

    def fit(
        self,
        endog: pd.DataFrame,
        exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        loss_q: Union[float, Iterable[float]] = 0.5,
        batch_size: int = 32,
        epochs: int = 100,
        shuffle: bool = True,
        early_stopping_patience: int = -1,
        validation_split: float = 0.0,
        sample_weights: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Self:
        """
        Train neural network by minimizing the quantile losses.

        May also run to update training of a pre-trained network, otherwise would build a new network and train it.

        Args:
            endog (pd.DataFrame): Target time series (i.e. signals).
            exog (Optional[Union[pd.DataFrame, List[pd.DataFrame]]], optional): eXogenous time series to fit targets. If
                not provided, the network will produce seasonal, temporal, and autoregressive-based forecasts only. If a
                List of dataframes is provided, then separte encoders and decoders will be built for each data frame.
                Defaults to None.
            loss_q (Union[float, Iterable[float]], optional): Quantile value for computing quantile loss. If the user
                attributes a similar cost to both overestimation and underestimation, then this value should be set
                close to 0.5 (i.e. attempts to fit to the median of the signal). Alternatively, user may provide an
                iterable of values to utilize different quantiles for the different decoders. Defaults to 0.5.
            batch_size (int, optional): Samples per batch to load. Defaults to 32.
            epochs (int, optional): Number of training passes. Defaults to 100.
            shuffle (bool, optional): Whether to reshuffle the data for every epoch. Defaults to True.
            early_stopping_patience (int, optional):  Number of epochs with no improvement after which training will be
                stopped. If passing through all epochs is desired, then this value should be set to -1. Defaults to -1.
            validation_split (float, optional): Ratio of data to be used for validation. If no validation required, this
                value should be 0.0. It is recommended to utilize validation specially when applying early stopping.
                Defaults to 0.0.
            sample_weights (Optional[pd.DataFrame], optional): Univariate time series of weights to bias the training
                samples. Defaults to None.
            **kwargs: Additional arguments to `BaseModel(...).fit` base model class methof.

        Returns:
            STARIXNet: Trained/refreshed neural network.
        """
        # inspect and register data/metadata
        endog, exog = super().fit(endog, exog=exog, **kwargs)

        # normalize
        endog, exog = self._preprocess_fit(endog, exog)._preprocess_transform(endog, exog)

        # build network (if not already built)
        if not self._params:
            self._freqs = self._extract_frequencies(endog)  # identify frequencies
            if self._endog_size == 1 and self.fit_multivariate:
                # TODO: add warning that dependencies cannot be modeled as multivariate when there is only 1 target
                self.fit_multivariate = False
            self._build()

        # convert to seq2seq (x and y)
        # TODO: this can be modified (includes_target) to fuse both past and future exog, when future exog are known
        x_time = self._convert_df_to_seq(
            pd.DataFrame(endog.index), self.forecast_steps, self.forecast_steps, includes_target=False
        )
        x_time = x_time.iloc[:-self.forecast_steps]  # to ensure size matches that of x_endog and y
        x_endog, y, y_sample_weights = self._convert_df_to_seq(
            endog, self.lagged_steps, self.forecast_steps, sample_weights=sample_weights
        )
        if exog is not None:
            if isinstance(exog, list):
                x_exog = [
                    self._convert_df_to_seq(_exog, self.lagged_steps, self.forecast_steps, includes_target=False)
                    for _exog in exog
                ]
                x_exog = [x.iloc[:-self.forecast_steps] for x in x_exog]  # ensures size matches with x_endog and y
            else:
                x_exog = self._convert_df_to_seq(
                    exog, self.lagged_steps, self.forecast_steps, includes_target=False
                )
                x_exog = x_exog.iloc[:-self.forecast_steps]  # to ensure size matches that of x_endog and y
        else:
            x_exog = None
            if 'exogenous' in self.applied_attributes:
                self.applied_attributes.remove('exogenous')  # avoid conflicts if exogenous is desired but no exog given
            
        # insure size matching
        if self.forecast_steps < self.lagged_steps:
            x_time = x_time.iloc[self.lagged_steps - self.forecast_steps:]
        elif self.forecast_steps > self.lagged_steps:
            diff = self.forecast_steps - self.lagged_steps
            x_endog, y = x_endog.iloc[diff:], y.iloc[diff:]
            if x_exog is not None:
                if isinstance(x_exog, list):
                    x_exog = [x.iloc[diff:] for x in x_exog]
                else:
                    x_exog = x_exog.iloc[diff:]
            if y_sample_weights is not None:
                y_sample_weights = y_sample_weights.iloc[diff:]

        # prepare pytorch dataloader
        xy_td_train, xy_td_val, kwargs = self._prepare_tensor_datasets(
            x_time, x_endog, x_exog, y, y_sample_weights, validation_split
        )
        xy_dataloader = DataLoader(xy_td_train, batch_size=batch_size, shuffle=shuffle,)

        # train
        if isinstance(loss_q, Iterable):
            criteria = [QLoss(q) for q in loss_q]
        else:
            criteria = [QLoss(loss_q) for decoder in self._decoder_labels]
        optimizer = torch.optim.Adam(self.parameters())
        trainer = Trainer(
            optimizer,
            self._forward,
            criteria,
            decoder_labels=self._decoder_labels,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
        )
        trainer.train(
            xy_dataloader, kwargs, xy_td_val, includes_sample_weights=True if sample_weights is not None else False
        )

        # validate model training
        for param in self.parameters():
            assert not torch.isnan(param).any().numpy(), (
                "Training corrupted: Found NaN value in neural network parameters. "
                "Please check all training inputs and configuration for errors"
            )
        return self

    def predict(
        self,
        endog: pd.DataFrame,
        exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        batch_size: int = 32,
        aggregated: bool = True,
        aggregation_q: float = 0.5,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Forecast future steps from input steps.

        Args:
            endog (pd.DataFrame): Input past time series of the target(s).
            exog (Optional[Union[pd.DataFrame, List[pd.DataFrame]]], optional): Input past time series of the exogenous
                signals. Defaults to None.
            batch_size (int, optional): Samples per batch to load. Currently, this is ignored. Defaults to 32.
            aggregated: (bool, optional): Whether to teturn an aggregated prediction across output decoders. Defaults to
                True.
            aggregation_q (float, optional): Quantile value for aggregating the decoder outputs. Defaults to 0.5.

        Returns:
            Union[np.ndarray, List[np.ndarray]]: If `aggregated` is True, return a 3d array of the aggregated forecast (
                no. of input sequences X future steps X no. of targets). Otherwise, return a list of the decoders'
                forecasts.
        """
        assert all([f in endog.columns for f in self._endog_labels]), (
            "one or more features are missing from the endog dataframe"
        )
        endog = endog[self._endog_labels]
        if exog is not None:
            if isinstance(exog, list):
                for i, _exog in enumerate(exog):
                    assert all([f in _exog.columns for f in self._exog_labels[i]]), (
                        f"one or more features are missing from the exog_{i} dataframe"
                    )
                    exog[i] = _exog[self._exog_labels[i]]
            else:
                assert all([f in exog.columns for f in self._exog_labels]), (
                    "one or more features are missing from the exog dataframe"
                )
                exog = exog[self._exog_labels]

        # normalize
        endog, exog = self._preprocess_transform(endog, exog)

        # convert to seq2seq (x and y)
        # TODO: this can be modified (includes_target) to fuse both past and future exog, when future exog are known
        x_time = self._convert_df_to_seq(
            pd.DataFrame(endog.index), self.forecast_steps, self.forecast_steps, includes_target=False
        )
        x_endog = self._convert_df_to_seq(endog, self.lagged_steps, self.forecast_steps, includes_target=False)
        if exog is not None:
            if isinstance(exog, list):
                x_exog = [
                    self._convert_df_to_seq(_exog, self.lagged_steps, self.forecast_steps, includes_target=False)
                    for _exog in exog
                ]
            else:
                x_exog = self._convert_df_to_seq(exog, self.lagged_steps, self.forecast_steps, includes_target=False)
        else:
            x_exog = None

        # insure size matching
        if self.forecast_steps < self.lagged_steps:
            x_time = x_time.iloc[self.lagged_steps - self.forecast_steps:]
        elif self.forecast_steps > self.lagged_steps:
            diff = self.lagged_steps - self.forecast_steps
            x_endog = x_endog.iloc[diff:]
            if x_exog:
                if isinstance(x_exog, list):
                    x_exog = [x.iloc[diff:] for x in x_exog]
                else:
                    x_exog = x_exog.iloc[diff:]

        # prepare pytorch dataloader
        x_td, _, kwargs = self._prepare_tensor_datasets(x_time, x_endog, x_exog)

        # obtain predictions
        y_preds = self._forward(x_td[:], kwargs)

        # inverse transform
        y_preds = [
            np.clip(
                (
                    (y_pred.detach().numpy() - self.minmaxscaler_range[0]) /
                    (self.minmaxscaler_range[1] - self.minmaxscaler_range[0])
                ) * (
                    self._endog_scalers[1] - self._endog_scalers[0]
                ).values + self._endog_scalers[0].values, 0, np.inf
            ) for y_pred in y_preds
        ]  # TODO: make this an _inverse_transform method and make it output dataframes as the input

        # aggregate predictions
        if aggregated:
            return np.quantile(y_preds, aggregation_q, axis=0)  # returns numpy of shape samples:timesteps:targets
        else:
            return y_preds  # TODO: this is a list of 3d tensors, need to figure out the different ways users may desire

    def get_config(self) -> Dict[str, Any]:
        """
        Extract SARIXNet configuration (i.e. parameters, variables, attributes, and settings).

        Returns:
            Dict[str, Any]: Dictionary of all instance variables and their values.
        """
        config = super().get_config()
        config.update(
            {
                'n_temporal_embeddings': self.n_temporal_embeddings,
                'embedding_dim': self.embedding_dim,
                'enforced_periods': self.enforced_periods,
                'max_periods': self.max_periods,
                'min_peak_distance': self.min_peak_distance,
                'min_oscillations': self.min_oscillations,
                'n_temporal_hidden_layers': self.n_temporal_hidden_layers,
                'n_ar_hidden_layers': self.n_ar_hidden_layers,
                'n_exog_hidden_layers': self.n_exog_hidden_layers,
                'trainable_freqs': self.trainable_freqs,
                'fit_multivariate': self.fit_multivariate,
                "endog_filters": self.endog_filters,
                "endog_filters_growth": self.endog_filters_growth,
                "endog_kernel_size": self.endog_kernel_size,
                'exog_filters': self.exog_filters,
                "exog_filters_growth": self.exog_filters_growth,
                'exog_kernel_size': self.exog_kernel_size,
                'applied_attributes': self.applied_attributes,
                '_time_scaler': self._time_scaler,
                '_freqs': self._freqs,
                '_params': self._params,
                '_moduls': self._moduls,
                '_forward_logic': self._forward_logic,
                '_decoder_labels': self._decoder_labels,
            }
        )
        return config

    @classmethod
    def load_model(cls, filepath: Union[str, BytesIO],) -> Self:
        """
        Load STARIXnet from file or memory bytes.

        Args:
            filepath (Union[str, BytesIO]): Path of stored model file. Can also be memory bytes.

        Returns:
            SARIXNet: Loaded neural network.
        """
        file = load(filepath)
        cls = cls(0)
        for param, val in file.items():
            setattr(cls, param, val)
        return cls

    def _extract_frequencies(self, endog: pd.DataFrame) -> List[float]:
        # identify frequencies
        freqs = extract_frequencies(
            endog, distance=self.min_peak_distance, min_periods=self.min_oscillations, max_freqs=self.max_periods
        )
        if self.enforced_periods:
            for ep in self.enforced_periods:
                ef = 2 * np.pi * (1 / ep)
                if ef not in freqs:
                    print(f'Enforcing a period of length {ep}')
                    freqs = [ef] + freqs
        return freqs

    def _convert_df_to_seq(
        self,
        df: pd.DataFrame,
        lagged_steps: int,
        forecast_steps: int,
        includes_target: bool = True,
        sample_weights: Optional[pd.DataFrame] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame]]:
        return ts2xyseq(
            df,
            lagged_steps=lagged_steps,
            forecast_steps=forecast_steps,
            lagged_features=df.columns.to_list(),
            skip_current_step=self.skip_current_step if includes_target else False,
            return_x_only=not includes_target,
            ts_weights=sample_weights
        )

    def _prepare_tensor_datasets(
        self,
        x_time: pd.DataFrame,
        x_endog: pd.DataFrame,
        x_exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        y: Optional[pd.DataFrame] = None,
        y_sample_weights: Optional[pd.DataFrame] = None,
        validation_split: float = 0.0,
    ) -> Tuple[TensorDataset, Optional[TensorDataset], List[Dict[str, Any]]]:
        if validation_split > 0.0:
            cutoff = round(validation_split * x_endog.shape[0])
            x_time_val, x_endog_val, y_val = x_time.iloc[-cutoff:], x_endog.iloc[-cutoff:], y.iloc[-cutoff:]
            x_time, x_endog, y = x_time.iloc[:-cutoff:], x_endog.iloc[:-cutoff], y.iloc[:-cutoff]
            tensors_val = [
                torch.tensor(x_time_val.values.reshape(-1, self.forecast_steps, 1)).float(),
                torch.tensor(x_endog_val.values.reshape(-1, self.lagged_steps, self._endog_size)).float(),
            ]
            if x_exog is not None:
                if isinstance(x_exog, list):
                    for i, x in enumerate(x_exog):
                        x_val = x.iloc[-cutoff:]
                        x_exog[i] = x.iloc[:-cutoff]
                        tensors_val.append(
                            torch.tensor(x_val.values.reshape(-1, self.lagged_steps, self._exog_size[i])).float()
                        )
                else:
                    x_exog_val = x_exog.iloc[-cutoff:]
                    x_exog = x_exog.iloc[:-cutoff]
                    tensors_val.append(
                        torch.tensor(x_exog_val.values.reshape(-1, self.lagged_steps, self._exog_size)).float()
                        )
            tensors_val.append(
                torch.tensor(y_val.values.reshape(-1, self.forecast_steps, self._endog_size)).float()
            )
            if y_sample_weights is not None:
                y_sample_weights_val = y_sample_weights.iloc[-cutoff:]
                y_sample_weights = y_sample_weights.iloc[:-cutoff]
                tensors_val.append(
                    torch.tensor(y_sample_weights_val.values.reshape(-1, self.forecast_steps, 1)).float()
                )
            xy_td_val = TensorDataset(*tensors_val)
        else:
            xy_td_val = None
        tensors = [
            torch.tensor(x_time.values.reshape(-1, self.forecast_steps, 1)).float(),
            torch.tensor(x_endog.values.reshape(-1, self.lagged_steps, self._endog_size)).float(),
        ]
        # kwargs = [{}, {}, {}]
        kwargs = {
            'seasonal': {}, 'temporal': {}, 'autoregressive': {}
        }  # additional arguments to pass with tesnors and tensors_val (for seasonal, temporal, ar)
        if x_exog is not None:
            if isinstance(x_exog, list):
                for i, x in enumerate(x_exog):
                    tensors.append(torch.tensor(x.values.reshape(-1, self.lagged_steps, self._exog_size[i])).float())
                    kwargs[f"exogenous_{i + 1}"] = {'suffix': f"_{i + 1}"}
            else:
                tensors.append(torch.tensor(x_exog.values.reshape(-1, self.lagged_steps, self._exog_size)).float())
                kwargs['exogenous'] = {}
        if y is not None:
            tensors.append(torch.tensor(y.values.reshape(-1, self.forecast_steps, self._endog_size)).float())
            kwargs['sample_weights'] = {}
        if y_sample_weights is not None:
            tensors.append(torch.tensor(y_sample_weights.values.reshape(-1, self.forecast_steps, 1)).float())
        xy_td = TensorDataset(*tensors)
        return xy_td, xy_td_val, kwargs

    def _build_seasonality_autoencoder(self) -> Self:
        if self._freqs:
            n_freqs = len(self._freqs)
            if self.trainable_freqs:
                self._params['freqs'] = nn.Parameter(torch.tensor(self._freqs))
            else:
                self._params['freqs'] = nn.Parameter(torch.tensor(self._freqs), requires_grad=False)
        else:
            n_freqs = self.max_periods
            self._params['freqs'] = nn.Parameter(torch.abs(torch.randn(n_freqs)))
        self._params['amps'] = nn.Parameter(torch.zeros(n_freqs, self._endog_size))  # amplitudes
        self._params['phases'] = nn.Parameter(torch.zeros(n_freqs, self._endog_size))  # angular freqs * phase shifts
        self._params['disps'] = nn.Parameter(torch.zeros(self._endog_size))  # displacements
        self._forward_logic['seasonal'] = self._forward_seasonality
        self._decoder_labels.append('seasonal')
        return self

    def _build_temporal_autoencoder(self) -> Self:
        if not self.embedding_dim:
            self.embedding_dim = min(50, (self.n_temporal_embeddings + 1) // 2)  # TODO: add refrence to this logic
        self._moduls['embds'] = nn.Embedding(self.n_temporal_embeddings, self.embedding_dim)  # temporal embeddings
        self._moduls['temporal_hlayers'] = nn.ModuleDict()  # for stacking temporal hiddel layers
        increment = (self.embedding_dim - self._endog_size) // (self.n_temporal_hidden_layers + 1)
        input_dim = self.embedding_dim
        for hl in range(self.n_temporal_hidden_layers):
            output_dim = input_dim - increment
            self._moduls['temporal_hlayers'][f'linear_{hl+1}'] = nn.Linear(input_dim, output_dim)
            input_dim = output_dim
        self._moduls['temporal_output'] = nn.Linear(input_dim, self._endog_size)  # output layer
        self._forward_logic['temporal'] = self._forward_temporal
        self._decoder_labels.append('temporal')
        return self

    def _build_autoregression_autoencoder(self) -> Self:
        # TODO: add option to use LSTMs
        # TODO: make type of pooling optional input
        self._moduls['ar_hlayers'] = nn.ModuleDict()  # for stacking AR hidden layers
        input_dim = self.lagged_steps - 1  # -1 for the differencing
        output_dim = self.forecast_steps
        increment = (output_dim - input_dim) // (self.n_ar_hidden_layers + 1)
        if self.fit_multivariate:
            # TODO: add option to use MLP instead of CNN to reduce model size further
            # TODO: add option to use hybrid CNN + MLP (i.e. CNN->Flatten->MLP->reshape)
            if not self.endog_filters:
                self.endog_filters = self._endog_size
            if not self.endog_kernel_size:
                self.endog_kernel_size = min(50, (input_dim + 1) // 2)  # TODO: verify that it is a good default
            input_filters = self._endog_size
            output_filters = self.endog_filters
            for hl in range(self.n_ar_hidden_layers):
                output_dim = input_dim + increment
                self._moduls['ar_hlayers'][f'conv_{hl+1}'] = nn.Conv1d(
                    input_filters, output_filters, self.endog_kernel_size
                )  # output will be filters x kernel_size (new_steps)
                self._moduls['ar_hlayers'][f'pool_{hl+1}'] = nn.AdaptiveAvgPool1d(output_dim)
                input_filters = output_filters
                output_filters = output_filters * self.endog_filters_growth
            self._moduls['ar_hlayers'][f'conv_{hl+2}'] = nn.Conv1d(
                input_filters, self._endog_size, self.endog_kernel_size
            )
            self._moduls['ar_output'] = nn.AdaptiveAvgPool1d(self.forecast_steps)
            self._forward_logic['autoregressive'] = self._forward_autoregression_multivariate
        else:
            for hl in range(self.n_ar_hidden_layers):
                output_dim = input_dim + increment
                self._moduls['ar_hlayers'][f'linear_{hl+1}'] = nn.ModuleList(
                    [nn.Linear(input_dim, output_dim) for _ in range(self._endog_size)]
                )
                input_dim = output_dim
            self._moduls['ar_output'] = nn.ModuleList(
                [nn.Linear(input_dim, self.forecast_steps) for _ in range(self._endog_size)]
            )  # output layer
            self._forward_logic['autoregressive'] = self._forward_autoregression_univariate
        self._decoder_labels.append('autoregressive')
        return self

    def _build_exogenous_autoencoder(
        self, exog_size: int, exog_filters: int, increment_kernel: int, increment_filters: int = 0, suffix: str = ''
    ) -> Self:
        # TODO: add option to use LSTMs
        # TODO: make type of pooling optional input
        self._moduls[f"exog{suffix}_hlayers"] = nn.ModuleDict()  # for stacking exogenous hidden layers
        if self.fit_multivariate:
            input_filters = exog_size
            output_filters = exog_filters
            for hl in range(self.n_exog_hidden_layers):
                output_dim = self.lagged_steps + increment_kernel
                self._moduls[f"exog{suffix}_hlayers"][f'conv_{hl+1}'] = nn.Conv1d(
                    input_filters, output_filters, self.exog_kernel_size
                )  # output will be filters x kernel_size (new_steps)
                self._moduls[f"exog{suffix}_hlayers"][f'pool_{hl+1}'] = nn.AdaptiveMaxPool1d(output_dim)
                input_filters = output_filters
                output_filters = (output_filters + increment_filters) * self.exog_filters_growth
            self._moduls[f"exog{suffix}_hlayers"][f'conv_{hl+2}'] = nn.Conv1d(
                input_filters, self._endog_size, self.exog_kernel_size
            )
            self._moduls[f"exog{suffix}_output"] = nn.AdaptiveMaxPool1d(self.forecast_steps)
            self._forward_logic[f"exogenous{suffix}"] = self._forward_exogenous_multivariate
        else:
            if exog_size > 1:  # TODO: a nd add condition for user to desire convolutions
                self._moduls[f"exog{suffix}_conv"] = nn.Conv1d(exog_size, exog_filters, self.exog_kernel_size)  # output will be filters x kernel_size (new_steps)
                self._moduls[f"exog{suffix}_pool"] = nn.MaxPool1d(max(1, self.exog_kernel_size // self.forecast_steps))  # TODO: need to verify these defaults are good
                input_dim = self.lagged_steps - 1 * (self.exog_kernel_size - 1)  # output from conv layer, assumes default values for padding (0), dilation (1), and stride (1)
                input_dim = exog_filters * (
                    (
                        (
                            input_dim - 1 * (self._moduls[f"exog{suffix}_pool"].kernel_size - 1) - 1
                        ) // self._moduls[f"exog{suffix}_pool"].stride
                    ) + 1
                )  # output from pooling, assumes default values for padding (0), dilation (1), and stride (kernel_size)
                output_dim = self.forecast_steps
            else:
                input_dim = self.lagged_steps * exog_size
            for hl in range(self.n_exog_hidden_layers):
                output_dim = input_dim + increment_kernel
                self._moduls[f"exog{suffix}_hlayers"][f'linear_{hl+1}'] = nn.ModuleList(
                    [nn.Linear(input_dim, output_dim) for _ in range(self._endog_size)]
                )
                input_dim = output_dim
            self._moduls[f"exog{suffix}_output"] = nn.ModuleList(
                [nn.Linear(input_dim, self.forecast_steps) for _ in range(self._endog_size)]
            )  # output layer
            self._forward_logic[f"exogenous{suffix}"] = self._forward_exogenous_univariate
        self._decoder_labels.append(f"exogenous{suffix}")
        return self

    def _forward_seasonality(self, x_t: torch.Tensor) -> torch.Tensor:
        y_s = x_t.reshape(-1, 1)  # samples:steps:1 -> samples*steps:1
        y_s = torch.mul(torch.unsqueeze(self._params['freqs'], 0), y_s)  # 1:freqs x samples*steps:1 -> samples*steps:freqs
        y_s = torch.add(torch.unsqueeze(y_s, -1), torch.unsqueeze(self._params['phases'], 0))  # samples*steps:freqs:1 + 1:freqs:outputs -> samples*steps:freqs:outputs
        y_s = torch.sum(torch.mul(torch.unsqueeze(self._params['amps'], 0), torch.sin(y_s)), 1)  # 1:freqs:outputs x samples*steps:freqs:outputs -> samples*steps:freqs:outputs -> samples*steps:outputs
        y_s = torch.add(y_s, torch.unsqueeze(self._params['disps'], 0))
        return y_s.reshape(-1, self.forecast_steps, self._endog_size)

    def _forward_temporal(self, x_t: torch.Tensor) -> torch.Tensor:
        y_t = x_t.int().reshape(-1, 1)  # samples:steps:1 -> samples*steps:1
        y_t = self._moduls['embds'](y_t % self.n_temporal_embeddings)  # samples*steps:1 -> samples*steps:1:embeddings
        y_t = self._flatten(y_t)  # samples*steps:1:embeddings -> samples*steps:embeddings
        for hl in range(self.n_temporal_hidden_layers):
            y_t = self._moduls['temporal_hlayers'][f'linear_{hl+1}'](y_t)
            y_t = self._relu(y_t)
        y_t = self._moduls['temporal_output'](y_t)
        return y_t.reshape(-1, self.forecast_steps, self._endog_size)

    def _forward_autoregression_univariate(self, x_a: torch.Tensor) -> torch.Tensor:
        y_a = (x_a[:, 1:, :] - x_a[:, :-1, :])
        y_a = list(torch.split(y_a, 1, dim=-1))
        y_a = [self._flatten(y_i) for y_i in y_a]
        for hl in range(self.n_ar_hidden_layers):
            for i, hl_i in enumerate(self._moduls['ar_hlayers'][f'linear_{hl+1}']):
                y_a[i] = hl_i(y_a[i])
                y_a[i] = self._relu(y_a[i])
            for i, o_i in enumerate(self._moduls['ar_output']):
                y_a[i] = o_i(y_a[i])
                y_a[i] = torch.unsqueeze(y_a[i], -1)
        y_a = torch.cat(y_a, dim=-1)  # samples:steps:outputs
        y_a = torch.cat((torch.unsqueeze(x_a[:, -1, :] + y_a[:, 0, :], 1), y_a[:, 1:, :]), dim=1)  # samples:steps:outputs
        return torch.cumsum(y_a, 1)

    def _forward_autoregression_multivariate(self, x_a: torch.Tensor) -> torch.Tensor:
        y_a = (x_a[:, 1:, :] - x_a[:, :-1, :])
        y_a = y_a.transpose(1, 2)  # samples:steps:inputs -> samples:inputs:steps
        for hl in range(self.n_ar_hidden_layers):
            y_a = self._moduls['ar_hlayers'][f'conv_{hl+1}'](y_a)  # samples:inputs:steps -> samples:filters:new_steps
            y_a = self._moduls['ar_hlayers'][f'pool_{hl+1}'](y_a)  # samples:filters:new_steps -> samples:filters:pooled_steps
        y_a = self._moduls['ar_hlayers'][f'conv_{hl+2}'](y_a)  # samples:filters:pooled_steps -> samples:outputs:new_steps
        y_a = self._moduls['ar_output'](y_a)  # samples:outputs:new_steps -> samples:outputs:steps
        y_a = y_a.transpose(1, 2)  # samples:outputs:steps -> samples:steps:outputs
        y_a = torch.cat((torch.unsqueeze(x_a[:, -1, :] + y_a[:, 0, :], 1), y_a[:, 1:, :]), dim=1)  # samples:steps:outputs
        return torch.cumsum(y_a, 1)

    def _forward_exogenous_univariate(self, x_e: torch.Tensor, suffix: str = '') -> torch.Tensor:
        if x_e.shape[-1] > 1:
            y_e = x_e.transpose(1, 2)  # samples:steps:inputs -> samples:inputs:steps
            y_e = self._moduls[f"exog{suffix}_conv"](y_e)  # samples:inputs:steps -> samples:filters:new_steps
            y_e = self._relu(y_e)
            y_e = self._moduls[f"exog{suffix}_pool"](y_e)  # samples:filters:new_steps -> samples:filters:pooled_steps
            y_e = y_e.transpose(1, 2)
            y_e = [self._flatten(y_e) for i in range(self._endog_size)]
        else:
            y_e = [self._flatten(x_e) for i in range(self._endog_size)]
        for hl in range(self.n_exog_hidden_layers):
            for i, hl_i in enumerate(self._moduls[f"exog{suffix}_hlayers"][f'linear_{hl+1}']):
                y_e[i] = hl_i(y_e[i])
                y_e[i] = self._relu(y_e[i])
        for i, o_i in enumerate(self._moduls[f"exog{suffix}_output"]):
            y_e[i] = o_i(y_e[i])
            y_e[i] = torch.unsqueeze(y_e[i], -1)
        return torch.cat(y_e, dim=-1)

    def _forward_exogenous_multivariate(self, x_e: torch.Tensor, suffix: str = '') -> torch.Tensor:
        y_e = x_e.transpose(1, 2)  # samples:steps:inputs -> samples:inputs:steps
        for hl in range(self.n_exog_hidden_layers):
            y_e = self._moduls[f"exog{suffix}_hlayers"][f'conv_{hl+1}'](y_e)  # samples:inputs:steps -> samples:filters:new_steps
            y_e = self._relu(y_e)
            y_e = self._moduls[f"exog{suffix}_hlayers"][f'pool_{hl+1}'](y_e)  # samples:filters:new_steps -> samples:filters:pooled_steps
        y_e = self._moduls[f"exog{suffix}_hlayers"][f'conv_{hl+2}'](y_e)  # samples:filters:pooled_steps -> samples:outputs:new_steps
        y_e = self._moduls[f"exog{suffix}_output"](y_e)  # samples:outputs:new_steps -> samples:outputs:steps
        return y_e.transpose(1, 2)  # samples:outputs:steps -> samples:steps:outputs
