#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self  # TODO: if only using py311 or later, Self can be in base typing
from zipfile import ZIP_DEFLATED, ZipFile

from joblib import dump, load
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import (
    AutoRegressiveBaseModelWithCovariates,
    DeepAR,
    DistributionLoss,
    MultiLoss,
    NormalDistributionLoss,
    QuantileLoss,
    RecurrentNetwork,
    RMSE,
    TemporalFusionTransformer,
    TimeSeriesDataSet
)
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
import torch

from starixnet.models import BaseModel


class PFBaseModel(BaseModel):
    def __init__(self, **kwargs):
        """
        Base model class for `pytorch-forecasting` auto-regressive models.

        Args:
            **kwargs: Additional arguments to pass to `starixnet.models.BaseModel` base model class.
        """
        super(PFBaseModel, self).__init__(**kwargs)
        self.time_varying_unknown_reals: Optional[List[str]] = None
        self._model: Optional[AutoRegressiveBaseModelWithCovariates] = None

        # hack for skipping a buffer time-step during prediction
        self._prediction_length = self.forecast_steps + 1 if self.skip_current_step else self.forecast_steps
        # TODO: Ideally modify the loss-fn to skipp the buffer time-step

        # hack for saving the model and loading it from dataset
        self._dataset: Optional[TimeSeriesDataSet] = None

    def fit(
        self,
        endog: pd.DataFrame,
        exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        batch_size: int = 32,
        epochs: int = 100,
        shuffle: bool = True,
        early_stopping_patience: int = -1,
        validation_split: float = 0.0,
        sample_weights: Optional[pd.DataFrame] = None,
        exog_as_unknown_reals: bool = False,
        gradient_clip_val: Optional[float] = None,
        **kwargs,
    ) -> Tuple[TimeSeriesDataSet, DataLoader, Optional[DataLoader], Trainer]:
        """
        Perform preliminary preparations for model training.

        Args:
            endog (pd.DataFrame): Target time series (i.e. signals).
            exog (Optional[Union[pd.DataFrame, List[pd.DataFrame]]], optional): eXogenous time series to fit targets. If
                not provided, the network will produce seasonal, temporal, and autoregressive-based forecasts only. If a
                List of dataframes is provided, then separte encoders and decoders will be built for each data frame.
                Defaults to None.
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
            exog_as_unknown_reals (bool, optional): Whether to set `exog` as the `time_varying_unknown_reals` parameter
                in the `TimeSeriesDataSet` object from the `pytorch_forecasting` library. Note that this works only for
                some models. Defaults to False
            gradient_clip_val (Optional[float], optional): The value at which to clip gradients. Defaults to None.

        Returns:
            (Tuple[pd.DataFrame, Optional[pd.DataFrame]]): Input time series after perofrming optional checks.
        """
        # inspect and register data/metadata
        endog, exog = super().fit(endog, exog=exog, **kwargs)

        # normalize
        endog, exog = self._preprocess_fit(endog, exog)._preprocess_transform(endog, exog)

        # format columns
        endog, exog = PFBaseModel._format_columns(endog, exog)

        # prepare dataset
        self.time_varying_unknown_reals = endog.columns.to_list()
        if exog is not None:
            if isinstance(exog, list):
                data = pd.concat([endog] + exog, axis=1)
                if exog_as_unknown_reals:
                    self.time_varying_unknown_reals += [c for _exog in exog for c in _exog.columns]
            else:
                data = pd.concat([endog, exog], axis=1)
                if exog_as_unknown_reals:
                    self.time_varying_unknown_reals += exog.columns.to_list()
        else:
            data = endog
        data['timestamp'] = endog.index  # this is already scaled during transformations
        data['constant'] = 1
        data = data.reset_index(drop=True)  # in case the index name conflicts with a column name

        # prepare pytorch dataloader
        if validation_split > 0.0:
            cutoff = data['timestamp'].iloc[round(validation_split * endog.shape[0])]
        else:
            cutoff = data['timestamp'].iloc[-1]
        training = TimeSeriesDataSet(
            data[lambda x: x['timestamp'] <= cutoff],
            'timestamp',  # column of time index
            endog.columns.to_list(),   # columns of prediction targets
            ['constant'],  # group ids identifying multiple time-series instances (for long dataframe)
            max_encoder_length=self.lagged_steps,
            max_prediction_length=self._prediction_length,
            time_varying_known_reals=['timestamp'],
            time_varying_unknown_reals=self.time_varying_unknown_reals,
        )
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, shuffle=shuffle)
        if validation_split > 0.0:
            validation = TimeSeriesDataSet.from_dataset(training, data, predict=True,)
            val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, shuffle=shuffle)
        else:
            val_dataloader = None

        # store minimal dataset sample for saving and loading
        self._dataset = TimeSeriesDataSet(
            data.iloc[-(self.lagged_steps + self._prediction_length):],
            'timestamp',  # column of time index
            endog.columns.to_list(),   # columns of prediction targets
            ['constant'],  # group ids identifying multiple time-series instances (for long dataframe)
            max_encoder_length=self.lagged_steps,
            max_prediction_length=self._prediction_length,
            time_varying_known_reals=['timestamp'],
            time_varying_unknown_reals=self.time_varying_unknown_reals,
        )  # TODO: repetitive function, should be abstracted

        # prepare trainer
        if early_stopping_patience > 0:
            callbacks = [
                EarlyStopping(
                    monitor="val_loss", min_delta=1e-4, patience=early_stopping_patience, verbose=False, mode="min"
                )
            ]
        else:
            callbacks = False
        trainer = Trainer(
            accelerator="cpu", callbacks=callbacks, max_epochs=epochs, gradient_clip_val=gradient_clip_val,
        )

        return training, train_dataloader, val_dataloader, trainer

    def predict(
        self, endog: pd.DataFrame, exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None, batch_size: int = 32,
    ) -> np.ndarray:
        """
        Forecast future steps from input steps.

        Args:
            endog (pd.DataFrame): Input past time series of the target(s).
            exog (Optional[Union[pd.DataFrame, List[pd.DataFrame]]], optional): Input past time series of the exogenous
                signals. Defaults to None.
            batch_size (int, optional): Samples per batch to load. Currently, this is ignored. Defaults to 32.

        Returns:
            np.ndarray: Aggregated forecasts in a 3d array: (no. of input sequences X future steps X no. of targets).
        """
        predict_dataloader = self._get_predict_dataloader(endog, exog=exog, batch_size=batch_size)

        # obtain predictions
        y_preds = [torch.Tensor() for target in range(self._endog_size)]
        for x, y in predict_dataloader:
            y_preds_temp = self._model.forward(x)['prediction']
            y_preds = [
                torch.cat((y_pred, y_pred_temp[:, -self.forecast_steps:, :]), dim=0)
                for y_pred, y_pred_temp in zip(y_preds, y_preds_temp)
            ]  # returns list of (samples:timesteps:target) for each target

        return self._postprocess_predictions(y_preds)

    def save(self, filepath: Union[str, BytesIO], compress: bool = True,) -> None:
        """
        Save model to file or memory bytes.

        Args:
            filepath (Union[str, BytesIO]): Path for storing model file. Can also be stored in memory bytes.
            compress (bool, optional): Whether to compress and reduce file size. Defaults to True.
        """
        # create in-memory zip file
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as zipf:
            # save config file
            config_file = self.get_config()
            config_file_bytes = BytesIO()  # create memory bytes
            dump(config_file, config_file_bytes, compress=compress)  # hold in memory
            zipf.writestr("config_file.z", config_file_bytes.getvalue())

            # save model state
            model_state_bytes = BytesIO()  # create memory bytes
            torch.save(self._model.state_dict(), model_state_bytes)  # hold in memory
            zipf.writestr("model_state.z", model_state_bytes.getvalue())

        # Save zipped archive from memory
        dump(zip_buffer.getvalue(), filepath, compress=compress)

    def get_config(self) -> Dict[str, Any]:
        """
        Extract model configuration (i.e. parameters, variables, attributes, and settings).

        Returns:
            Dict[str, Any]: Dictionary of all instance variables and their values.
        """
        config = super().get_config()
        config.update(
            {
                "time_varying_unknown_reals": self.time_varying_unknown_reals,
                "_prediction_length": self._prediction_length,
                "_dataset": self._dataset,
            }
        )
        return config

    @staticmethod
    def _format_columns(
        endog: pd.DataFrame, exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ) -> Tuple[pd.DataFrame, Optional[Union[pd.DataFrame, List[pd.DataFrame]]]]:
        if isinstance(endog.columns, pd.core.indexes.multi.MultiIndex):
            endog.columns = endog.columns.map('&'.join)
            if exog is not None:
                if isinstance(exog, list):
                    for _exog in exog:
                        _exog.columns = _exog.columns.map('&'.join)
                else:
                    exog.columns = exog.columns.map('&'.join)
        return endog, exog

    def _param_check(self):
        for param in self.parameters():
            assert not torch.isnan(param).any().numpy(), (
                "Training corrupted: Found NaN value in neural network parameters. "
                "Please check all training inputs and configuration for errors"
            )

    def _get_predict_dataloader(
        self, endog: pd.DataFrame, exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None, batch_size: int = 32,
    ) -> DataLoader:
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

        # format columns
        endog, exog = PFBaseModel._format_columns(endog, exog)

        # prepare encoder dataset
        if exog is not None:
            if isinstance(exog, list):
                encoder_data = pd.concat([endog] + exog, axis=1)
            else:
                encoder_data = pd.concat([endog, exog], axis=1)
        else:
            encoder_data = endog
        encoder_data['timestamp'] = endog.index  # this is already scaled during transformations
        encoder_data['constant'] = 1  # TODO: shouldn't be constant if multiple separate predictions are needed
        encoder_data = encoder_data.reset_index(drop=True)  # in case the index name conflicts with a column name

        # create future time steps (decoder dataset)
        predict_timestamp = encoder_data['timestamp'].iloc[-1] + 1
        prediction_data = pd.DataFrame({
            "timestamp": range(predict_timestamp, predict_timestamp + self._prediction_length),
            'constant': 1
        })
        # return encoder_data, prediction_data

        # prepare pytorch dataloader
        combined = pd.concat([encoder_data, prediction_data], ignore_index=True).fillna(0.0)  # dummy 0 to avoid error
        prediction = TimeSeriesDataSet(
            combined,
            'timestamp',  # column of time index
            endog.columns.to_list(),   # columns of prediction targets
            ['constant'],  # group ids identifying multiple time-series instances (for long dataframe)
            max_encoder_length=self.lagged_steps,
            max_prediction_length=self._prediction_length,
            time_varying_known_reals=['timestamp'],
            time_varying_unknown_reals=self.time_varying_unknown_reals,
        )
        return prediction.to_dataloader(train=False, batch_size=batch_size, shuffle=False)

    def _postprocess_predictions(self, y_preds: List[torch.Tensor]) -> np.ndarray:
        # concatenate tensors to return (samples:timesteps:targets)
        y_preds = torch.cat(y_preds, dim=-1)

        # inverse transform
        y_preds = np.clip(
            (
                (y_preds.detach().cpu().numpy() - self.minmaxscaler_range[0]) /
                (self.minmaxscaler_range[1] - self.minmaxscaler_range[0])
            ) * (
                self._endog_scalers[1] - self._endog_scalers[0]
            ).values + self._endog_scalers[0].values, 0, np.inf
        )  # TODO: make this an _inverse_transform method and make it output dataframes as the input

        return y_preds


class LSTM(PFBaseModel):
    def __init__(self, hidden_size: int = 1, rnn_layers: int = 1, dropout: float = 0.0, **kwargs,):
        """
        LSTM followed by output layer.

        Args:
            hidden_size (int, optional): Hidden recurrent size. Source suggests 10. Defaults to 1.
            rnn_layers (int, optional): Number of RNN layers. Source suggests 2. Defaults to 1.
            dropout (float, optional): Dropout in RNN layers. Source siggests 0.1. Defaults to 0.0.
            **kwargs: Additional arguments to pass to `PFBaseModel` base model class.
        """
        super(LSTM, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout

        self._model: Optional[RecurrentNetwork] = None

    def _build(self, training: TimeSeriesDataSet,) -> Self:
        self._model = RecurrentNetwork.from_dataset(
            training,
            cell_type='LSTM',
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            loss=RMSE(),  # instead of QuantileLoss([loss_q])
            optimizer='adam',
        )
        return self

    def fit(self, *args, **kwargs) -> Self:
        # prepare for training data using  base model
        training, train_dataloader, val_dataloader, trainer = super().fit(*args, **kwargs)

        # build network (if not already built)
        if self._model is None:
            self._build(training)

        # train
        trainer.fit(self._model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,)

        # check
        self._param_check()

        return self

    def get_config(self) -> Dict[str, Any]:
        """
        Extract model configuration (i.e. parameters, variables, attributes, and settings).

        Returns:
            Dict[str, Any]: Dictionary of all instance variables and their values.
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "rnn_layers": self.rnn_layers,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def load_model(cls, filepath: Union[str, BytesIO],) -> Self:
        """
        Load LSTM from file or memory bytes.

        Args:
            filepath (Union[str, BytesIO]): Path of stored model file. Can also be memory bytes.

        Returns:
            LSTM: Loaded neural network.
        """
        # initialize empty class
        cls = cls(0)

        # load components from zipped file
        zipped_bytes = load(filepath)
        zip_buffer = BytesIO(zipped_bytes)
        with ZipFile(zip_buffer, "r") as zipf:
            # load and set config
            config_file = load(BytesIO(zipf.read("config_file.z")))
            for param, val in config_file.items():
                setattr(cls, param, val)

            # load model state and set model
            model_state = torch.load(BytesIO(zipf.read("model_state.z")))
            cls._model = RecurrentNetwork.from_dataset(
                cls._dataset,
                cell_type='LSTM',
                hidden_size=cls.hidden_size,
                rnn_layers=cls.rnn_layers,
                dropout=cls.dropout,
                loss=RMSE(),  # instead of QuantileLoss([loss_q])
                optimizer='adam',
            )  # TODO: repetitive function, should be abstracted
            cls._model.load_state_dict(model_state)

        return cls


class DeepVAR(PFBaseModel):
    def __init__(
        self,
        hidden_size: int = 1,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        loss: Union[DistributionLoss, MultiLoss] = NormalDistributionLoss(),  # TODO: make str to load from yaml
        **kwargs,
    ):
        """
        Deep (Variational) AutoRegressive Network.

        Args:
            hidden_size (int, optional): Hidden recurrent size. Source suggests 10. Defaults to 1.
            rnn_layers (int, optional): Number of RNN layers. Source suggests 2. Defaults to 1.
            dropout (float, optional): Dropout in RNN layers. Source siggests 0.1. Defaults to 0.0.
            loss (Union[DistributionLoss, MultiLoss], optional): Distribution loss function. By using a Multivariate
                Loss such as the `MultivariateNormalDistributionLoss`, the network is converted into a variational
                (DeepVAR) network. Otherwise, the network becomes a normal-probabilistic (DeepAR) network. Defaults to
                `NormalDistributionLoss()`.
            **kwargs: Additional arguments to pass to `PFBaseModel` base model class.
        """
        super(DeepVAR, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.loss = loss

        self._model: Optional[DeepAR] = None

    def _build(self, training: TimeSeriesDataSet,) -> Self:
        if self._endog_size > 1 or isinstance(training.target, (list, tuple)):
            self.loss = MultiLoss([self.loss] * self._endog_size)  # must provide loss for each target
        self._model = DeepAR.from_dataset(
            training,
            cell_type='LSTM',
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            loss=self.loss,
            optimizer='adam',
        )
        return self

    def fit(self, *args, **kwargs) -> Self:
        # prepare for training data using  base model
        training, train_dataloader, val_dataloader, trainer = super().fit(*args, **kwargs)

        # build network (if not already built)
        if self._model is None:
            self._build(training)

        # train
        trainer.fit(self._model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,)

        # check
        self._param_check()

        return self

    def predict(
        self, endog: pd.DataFrame, exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None, batch_size: int = 32,
    ) -> np.ndarray:
        """
        Forecast future steps from input steps.

        Args:
            endog (pd.DataFrame): Input past time series of the target(s).
            exog (Optional[Union[pd.DataFrame, List[pd.DataFrame]]], optional): Input past time series of the exogenous
                signals. Defaults to None.
            batch_size (int, optional): Samples per batch to load. Defaults to 32.

        Returns:
            np.ndarray: Aggregated forecasts in a 3d array: (no. of input sequences X future steps X no. of targets).
        """
        predict_dataloader = super()._get_predict_dataloader(endog, exog=exog, batch_size=batch_size)
        y_preds = self._model.predict(predict_dataloader, mode="prediction")
        y_preds = [y_pred[:, -self.forecast_steps:].unsqueeze(-1) for y_pred in y_preds]  # remove buffer and make 3d
        return super()._postprocess_predictions(y_preds)

    def get_config(self) -> Dict[str, Any]:
        """
        Extract model configuration (i.e. parameters, variables, attributes, and settings).

        Returns:
            Dict[str, Any]: Dictionary of all instance variables and their values.
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "rnn_layers": self.rnn_layers,
                "dropout": self.dropout,
                "loss": self.loss,
            }
        )
        return config

    @classmethod
    def load_model(cls, filepath: Union[str, BytesIO],) -> Self:
        """
        Load DeepVAR from file or memory bytes.

        Args:
            filepath (Union[str, BytesIO]): Path of stored model file. Can also be memory bytes.

        Returns:
            DeepVAR: Loaded neural network.
        """
        # initialize empty class
        cls = cls(0)

        # load components from zipped file
        zipped_bytes = load(filepath)
        zip_buffer = BytesIO(zipped_bytes)
        with ZipFile(zip_buffer, "r") as zipf:
            # load and set config
            config_file = load(BytesIO(zipf.read("config_file.z")))
            for param, val in config_file.items():
                setattr(cls, param, val)

            # load model state and set model
            model_state = torch.load(BytesIO(zipf.read("model_state.z")))
            cls._model = DeepAR.from_dataset(
                cls._dataset,
                cell_type='LSTM',
                hidden_size=cls.hidden_size,
                rnn_layers=cls.rnn_layers,
                dropout=cls.dropout,
                loss=cls.loss,
                optimizer='adam',
            )  # TODO: repetitive function, should be abstracted
            cls._model.load_state_dict(model_state)

        return cls


class TFT(PFBaseModel):
    def __init__(
        self,
        hidden_size: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        quantiles: List[float] = [0.5],
        **kwargs,
    ):
        """
        Temporal Fusion Transfomer.

        Args:
            hidden_size (int, optional): hidden size of network which can range from 8 to 512. Source suggests 16.
                Defaults to 1.
            lstm_layers (int, optional): Number of LSTM layers. Source suggests 2 as optimal. Defaults to 1.
            dropout (float, optional): Dropout rate. Source siggests 0.1. Defaults to 0.0.
            quantiles (List[float], optional): Quantiles for `pytorch_forecasting.QuantileLoss` loss function. Source
                suggests 7 quantiles (i.e. [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]). Defaults to [0.5].
            **kwargs: Additional arguments to pass to `PFBaseModel` base model class.
        """
        super(TFT, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.quantiles = quantiles
        self.output_size: Union[int, List[int]] = len(self.quantiles)

        self._model: Optional[TemporalFusionTransformer] = None

    def _build(self, training: TimeSeriesDataSet,) -> Self:
        self.output_size = len(self.quantiles) if isinstance(self.quantiles, (list, tuple)) else 1
        if self._endog_size > 1:
            self.output_size = [self.output_size] * self._endog_size
        self._model = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            output_size=self.output_size,
            loss=QuantileLoss(self.quantiles),
            optimizer='adam',
        )
        return self

    def fit(self, *args, **kwargs) -> Self:
        # prepare for training data using  base model
        training, train_dataloader, val_dataloader, trainer = super().fit(*args, **kwargs)

        # build network (if not already built)
        if self._model is None:
            self._build(training)

        # train
        trainer.fit(self._model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,)

        # check
        self._param_check()

        return self

    def get_config(self) -> Dict[str, Any]:
        """
        Extract model configuration (i.e. parameters, variables, attributes, and settings).

        Returns:
            Dict[str, Any]: Dictionary of all instance variables and their values.
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "lstm_layers": self.lstm_layers,
                "dropout": self.dropout,
                "quantiles": self.quantiles,
                "output_size": self.output_size,
            }
        )
        return config

    @classmethod
    def load_model(cls, filepath: Union[str, BytesIO],) -> Self:
        """
        Load TFT from file or memory bytes.

        Args:
            filepath (Union[str, BytesIO]): Path of stored model file. Can also be memory bytes.

        Returns:
            TFT: Loaded neural network.
        """
        # initialize empty class
        cls = cls(0)

        # load components from zipped file
        zipped_bytes = load(filepath)
        zip_buffer = BytesIO(zipped_bytes)
        with ZipFile(zip_buffer, "r") as zipf:
            # load and set config
            config_file = load(BytesIO(zipf.read("config_file.z")))
            for param, val in config_file.items():
                setattr(cls, param, val)

            # load model state and set model
            model_state = torch.load(BytesIO(zipf.read("model_state.z")))
            cls._model = TemporalFusionTransformer.from_dataset(
                cls._dataset,
                hidden_size=cls.hidden_size,
                lstm_layers=cls.lstm_layers,
                dropout=cls.dropout,
                output_size=cls.output_size,
                loss=QuantileLoss(cls.quantiles),
                optimizer='adam',
            )  # TODO: repetitive function, should be abstracted
            cls._model.load_state_dict(model_state)

        return cls


class SARIMA(BaseModel):  # TODO: if more statistical models will be used, then new basemodel class is needed
    def __init__(
        self,
        ar_order: int = 1,
        differencing: int = 1,
        ma_order: int = 0,
        seasonal_ar_order: int = 0,
        seasonal_differencing: int = 0,
        seasonal_ma_order: int = 0,
        season_length: int = 0,
        **kwargs,
    ):
        """
        Seasonal Autoregressive Integrated Moving Average models.

        Args:
            ar_order (int, optional): AutoRegressive (AR) terms (lag of past values). Defaults to 1.
            differencing (int, optional): Number of times the data is differenced to make it stationary. Defaults to 1.
            ma_order (int, optional): Moving Average (MA) terms (lag of past errors). Defaults to 0.
            seasonal_ar_order (int, optional): 	Seasonal AR terms (lags of the seasonal period). Defaults to 0.
            seasonal_differencing (int, optional): How many times to subtract seasonally lagged data. Defaults to 0.
            seasonal_ma_order (int, optional): Seasonal MA terms (seasonal lags of past errors). Defaults to 0.
            season_length (int, optional): Number of time steps per season. Defaults to 0.
            **kwargs (TYPE): Additional arguments to pass to `PFBaseModel` base model class.

        Returns:
            None.

        """
        super(SARIMA, self).__init__(**kwargs)
        self.order: Tuple[int] = (ar_order, differencing, ma_order)
        self.seasonal_order: Tuple[int] = (seasonal_ar_order, seasonal_differencing, seasonal_ma_order, season_length)

        self._model: Optional[Union[SARIMAX, List[SARIMAX]]] = None
        self._res: Optional[Union[SARIMAXResults, List[SARIMAXResults]]] = None
        self._state_timestamp: int = 0  # to record last training point from which ot start predicting from
        self._state_size: int = 0  # to record the size of the training data used as the state
        self._step_size: int = 0
        
        # hack for skipping a buffer time-step during prediction
        self._prediction_length = self.forecast_steps + 1 if self.skip_current_step else self.forecast_steps

    def _build(self, endog: pd.DataFrame) -> Self:
        kwargs = {'order': self.order, 'seasonal_order': self.seasonal_order, "simple_differencing": False}
        if self._endog_size > 1:
            self._model = [SARIMAX(endog[label].values.reshape(-1), **kwargs,) for label in self._endog_labels]
        else:
            self._model = SARIMAX(endog.values.reshape(-1), **kwargs,)
        return self

    def fit(
        self,
        endog: pd.DataFrame,
        exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        epochs: int = 100,
        low_memory: bool = True,
        max_state_size: Optional[int] = None,  # to limit the saved model size
        **kwargs,
    ) -> Self:
        # ignore exog for this kind of model
        exog = None  # TODO: potentially offset exog and use as external regressor

        # inspect and register data/metadata
        endog, exog = super().fit(endog, exog=exog, **kwargs)
        self._state_timestamp = endog.index[-1].item()
        endog_size = endog.shape[0]
        if max_state_size:
            self._state_size = max_state_size if endog_size > max_state_size else endog_size
        else:
            self._state_size = endog_size
        epochtime = endog.index.astype(int)[:2]
        self._step_size = epochtime[1] - epochtime[0]

        # build model (if not already built)
        if self._model is None:
            self._build(endog)

        # train
        if self._endog_size > 1:
            self._res = [
                self._fit(
                    model, epochs=epochs, low_memory=low_memory, max_state_size=max_state_size
                ) for model in self._model
            ]
        else:
            self._res = self._fit(self._model, epochs=epochs, low_memory=low_memory, max_state_size=max_state_size)

        return self

    def predict(
        self, endog: pd.DataFrame, exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ) -> np.ndarray:
        # ignore exog for this kind of model
        exog = None  # TODO: potentially offset exog and use as external regressor

        # update model states
        missing_state_size = int(((endog.index[0].item() - self._state_timestamp) // self._step_size) - 1)
        if self._endog_size > 1:
            self._res = [
                self._update_state(
                    res, missing_state_size, endog[label], exog=exog
                ) for res, label in zip(self._res, self._endog_labels)
            ]
        else:
            self._res = self._update_state(self._res, missing_state_size, endog, exog=exog)
        if endog.index[-1].item() > self._state_timestamp:
            self._state_timestamp = endog.index[-1].item()

        # generate and transform predictions
        if self._endog_size > 1:
            y_preds = [self._get_forecasts(res) for res in self._res]
            y_preds = np.concatenate(y_preds, axis=-1)
        else:
            y_preds = self._get_forecasts(self._re)

        return np.clip(y_preds, 0.0, np.inf)

    def get_config(self) -> Dict[str, Any]:
        """
        Extract model configuration (i.e. parameters, variables, attributes, and settings).

        Returns:
            Dict[str, Any]: Dictionary of all instance variables and their values.
        """
        config = super().get_config()
        config.update(
            {
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "_model": self._model,
                "_res": self._res,
                "_state_timestamp": self._state_timestamp,
                "_state_size": self._state_size,
                "_step_size": self._step_size,
                "_prediction_length": self._prediction_length,
            }
        )
        return config

    def _fit(
        self, model: SARIMAX, epochs: int = 100, low_memory: bool = True, max_state_size: Optional[int] = None
    ) -> SARIMAXResults:
        # train model
        res = model.fit(maxiter=epochs, low_memory=low_memory,)

        # reduce model size
        if max_state_size:
            res.data.endog = res.data.endog[-self._state_size:]
            res.data.data.orig_endog = res.data.data.orig_endog[-self._state_size:]
            res.states.filtered = res.states.filtered[-self._state_size:]
            res.states.filtered_cov = res.states.filtered_cov[-self._state_size:]
            res.states.predicted = res.states.predicted[-self._state_size:]
            res.states.predicted_cov = res.states.predicted_cov[-self._state_size:]
            res.states.smoothed = res.states.smoothed[-self._state_size:]
            res.states.smoothed_cov = res.states.smoothed_cov[-self._state_size:]
            res.filtered_state = res.filtered_state[:, -self._state_size + 1:]
            res.filtered_state_cov = res.filtered_state_cov[:, :, -self._state_size + 1:]
            res.predicted_state = res.predicted_state[:, -self._state_size:]
            res.predicted_state_cov = res.predicted_state_cov[:, :, -self._state_size:]
            res.forecasts = res.forecasts[:, -self._state_size + 1:]
            res.forecasts_error = res.forecasts_error[:, -self._state_size + 1:]
            res.forecasts_error_cov = res.forecasts_error_cov[:, :, -self._state_size + 1:]
            res.standardized_forecasts_error = res.standardized_forecasts_error[:, -self._state_size + 1:]
            res.scaled_smoothed_estimator = res.scaled_smoothed_estimator[:, -self._state_size + 1:]
            res.scaled_smoothed_estimator_cov = res.scaled_smoothed_estimator_cov[:, :, -self._state_size + 1:]
            res.smoothing_error = res.smoothing_error[:, -self._state_size + 1:]
            res.smoothed_state = res.smoothed_state[:, -self._state_size + 1:]
            res.smoothed_state_cov = res.smoothed_state_cov[:, :, -self._state_size + 1:]
            res.smoothed_state_autocov = res.smoothed_state_autocov[:, :, -self._state_size + 1:]
            res.smoothed_measurement_disturbance = res.smoothed_measurement_disturbance[:, -self._state_size + 1:]
            res.smoothed_state_disturbance = res.smoothed_state_disturbance[:, -self._state_size + 1:]
            res.smoothed_measurement_disturbance_cov = (
                res.smoothed_measurement_disturbance_cov[:, :, -self._state_size + 1:]
            )
            res.smoothed_state_disturbance_cov = res.smoothed_state_disturbance_cov[:, :, -self._state_size + 1:]
            # TODO: still need to browse and clearn res.filter_results
            # TODO: still need to browse and clearn res.smoother_results

        return res

    def _update_state(
        self,
        res: SARIMAXResults,
        missing_state_size: int,
        endog: Union[pd.DataFrame, pd.Series],
        exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ) -> SARIMAXResults:
        if missing_state_size > 0:
            missing_state = res.forecast(missing_state_size)  # fill missing gaps with forecasted values
            res = res.append(missing_state)  # add missing gaps
        elif missing_state_size < 0:  # in rare case of being recently trained that there is an overlap with endog
            endog = endog.iloc[-missing_state_size:]
        if endog.shape[0] > 0:
            res = res.append(endog.values.reshape(-1))   # add encoder context
        res.data.endog = res.data.endog[-self._state_size:]  # to avoid memory leak (model can keep increasing in mem)
        return res

    def _get_forecasts(self, res: SARIMAXResults,) -> np.ndarray:
        return res.forecast(self._prediction_length)[-self.forecast_steps:].reshape(1, -1, 1)

    @classmethod
    def load_model(cls, filepath: Union[str, BytesIO],) -> Self:
        """
        Load SARIMA from file or memory bytes.

        Args:
            filepath (Union[str, BytesIO]): Path of stored model file. Can also be memory bytes.

        Returns:
            SARIMA: Loaded Seasonal Autoregressive Integrated Moving Average models.
        """
        file = load(filepath)
        cls = cls(0)
        for param, val in file.items():
            setattr(cls, param, val)
        return cls
