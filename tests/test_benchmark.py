#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

from os import path, makedirs
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from benchmark.models import DeepVAR, LSTM, SARIMA, TFT

cur_dir = path.dirname(path.realpath(__file__))


class TestBenchMark(object):
    def tests(self):
        endog = pd.read_csv(path.join(cur_dir, "data", "endog.csv"), index_col=[0]).interpolate()
        exog_1 = pd.read_csv(path.join(cur_dir, "data", "exog_1.csv"), index_col=[0]).interpolate()
        exog_2 = pd.read_csv(path.join(cur_dir, "data", "exog_2.csv"), index_col=[0]).interpolate()
        self._test_lstm(endog, [exog_1, exog_2])
        self._test_deepar(endog, [exog_1, exog_2])  # TODO: potentially test deepvar as well as this deepar
        self._test_tft(endog, [exog_1, exog_2])
        self._test_sarima(endog)

    def _test_lstm(self, endog: pd.DataFrame, exog: Union[pd.DataFrame, List[pd.DataFrame]],):
        model = LSTM(lagged_steps=30, forecast_steps=15, skip_current_step=True,)
        assert not model._model  # test that there is no model built
        self.__test_fit(
            model, endog.copy(), exog.copy(), batch_size=256, epochs=3, early_stopping_patience=1, validation_split=0.3,
        )  # making copies because List data are mutable
        self.__test_predict(model, endog.iloc[-1440-45:-15], [_exog.iloc[-1440-45:-15] for _exog in exog])
        self.__test_save(model, 'lstm')
        self.__test_load_model(LSTM, 'lstm')

    def _test_deepar(self, endog: pd.DataFrame, exog: Union[pd.DataFrame, List[pd.DataFrame]],):
        model = DeepVAR(lagged_steps=30, forecast_steps=15, skip_current_step=True,)
        assert not model._model  # test that there is no model built
        self.__test_fit(
            model, endog.copy(), exog.copy(), batch_size=256, epochs=3, early_stopping_patience=1, validation_split=0.3,
        )  # making copies because List data are mutable
        self.__test_predict(model, endog.iloc[-1440-45:-15], [_exog.iloc[-1440-45:-15] for _exog in exog])
        self.__test_save(model, 'deepar')
        self.__test_load_model(DeepVAR, 'deepar')

    def _test_tft(self, endog: pd.DataFrame, exog: Union[pd.DataFrame, List[pd.DataFrame]],):
        model = TFT(lagged_steps=30, forecast_steps=15, skip_current_step=True,)
        assert not model._model  # test that there is no model built
        self.__test_fit(
            model,
            endog.copy(),
            exog.copy(),
            batch_size=256,
            epochs=3,
            early_stopping_patience=1,
            validation_split=0.3,
            exog_as_unknown_reals=True,
        )  # making copies because List data are mutable
        self.__test_predict(model, endog.iloc[-1440-45:-15], [_exog.iloc[-1440-45:-15] for _exog in exog])
        self.__test_save(model, 'tft')
        self.__test_load_model(TFT, 'tft')

    def _test_sarima(self, endog: pd.DataFrame):
        model = SARIMA(lagged_steps=30, forecast_steps=15, skip_current_step=True, ar_order=30, differencing=1,)
        assert not model._model  # test that there is no model built
        self.__test_fit(model, endog.copy(), epochs=3,)
        self.__test_predict(model, endog.iloc[-1440-45:-15])
        self.__test_save(model, 'sarima')
        self.__test_load_model(SARIMA, 'sarima')

    def __test_fit(
        self,
        model: Union[DeepVAR, LSTM, SARIMA, TFT],
        endog: pd.DataFrame,
        exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        **kwargs,
    ):
        model.fit(endog, exog=exog, **kwargs)
        assert model._model is not None  # test if model was set
        if not isinstance(model, SARIMA):
            assert len(model._endog_scalers) == 2 and len(model._exog_scalers) == 2  # test if scalers are extracted
            assert model._dataset is not None
            assert all(
                [exog_size > 0 for exog_size in model._exog_size]
            ) if isinstance(exog, list) else model._exog_size > 0
        else:
            assert model._res is not None
        assert model._endog_size > 0

    def __test_predict(
        self,
        model: Union[DeepVAR, LSTM, SARIMA, TFT],
        endog: pd.DataFrame,
        exog: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
    ):
        preds = model.predict(endog, exog)
        if not isinstance(model, SARIMA):
            assert preds.shape[0] == 1440 + model._prediction_length - model.forecast_steps
        else:
            assert preds.shape[0] == 1
        assert preds.shape[1] == model.forecast_steps and preds.shape[2] == model._endog_size
        assert preds.min() >= 0 and preds.max() < np.inf

    def __test_save(self, model: Union[DeepVAR, LSTM, SARIMA, TFT], filename: str):
        if not path.exists(path.join(cur_dir, "models")):
            makedirs(path.join(cur_dir, "models"))
        model.save(path.join(cur_dir, "models", f"{filename}.z"))
        assert path.exists(path.join(cur_dir, "models", f"{filename}.z"))

    def __test_load_model(self, model: Union[DeepVAR, LSTM, SARIMA, TFT], filename: str):
        model = model.load_model(path.join(cur_dir, "models", f"{filename}.z"))
        assert model._model is not None
        if not isinstance(model, SARIMA):
            assert model._dataset is not None
        else:
            assert model._res is not None
        # TODO: should ideally compare every attribute of loaded model with trained model
