"""Tests for the benchmark forecasting models.

Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory of this source
tree.
"""

from __future__ import annotations

from os import makedirs, path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.models import LSTM, SARIMA, TFT, DeepAR, DLinear, PatchTST, iTransformer

cur_dir = path.dirname(path.realpath(__file__))


class TestBenchMark:
    def tests(self) -> None:
        endog = pd.read_csv(path.join(cur_dir, "data", "endog.csv"), index_col=[0]).interpolate()
        exog_1 = pd.read_csv(path.join(cur_dir, "data", "exog_1.csv"), index_col=[0]).interpolate()
        exog_2 = pd.read_csv(path.join(cur_dir, "data", "exog_2.csv"), index_col=[0]).interpolate()
        self._test_lstm(endog, [exog_1, exog_2])
        self._test_deepar(endog, [exog_1, exog_2])
        self._test_tft(endog, [exog_1, exog_2])
        self._test_dlinear(endog, [exog_1, exog_2])
        self._test_patchtst(endog, [exog_1, exog_2])
        self._test_itransformer(endog, [exog_1, exog_2])
        self._test_sarima(endog)

    def _test_lstm(self, endog: pd.DataFrame, exog: list[pd.DataFrame]) -> None:
        model = LSTM(lagged_steps=30, forecast_steps=15, skip_current_step=True, hidden_size=8)
        assert model._model is None
        self.__test_fit(
            model,
            endog.copy(),
            exog.copy(),
            batch_size=256,
            epochs=3,
            early_stopping_patience=1,
            validation_split=0.3,
            exog_as_unknown_reals=True,
        )
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15], [e.iloc[-1440 - 45 : -15] for e in exog])
        self.__test_save_and_load(model, LSTM, "lstm")

    def _test_deepar(self, endog: pd.DataFrame, exog: list[pd.DataFrame]) -> None:
        model = DeepAR(lagged_steps=30, forecast_steps=15, skip_current_step=True, hidden_size=8)
        assert model._model is None
        self.__test_fit(
            model,
            endog.copy(),
            exog.copy(),
            batch_size=256,
            epochs=3,
            early_stopping_patience=1,
            validation_split=0.3,
            exog_as_unknown_reals=True,
        )
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15], [e.iloc[-1440 - 45 : -15] for e in exog])
        self.__test_save_and_load(model, DeepAR, "deepar")

    def _test_tft(self, endog: pd.DataFrame, exog: list[pd.DataFrame]) -> None:
        model = TFT(lagged_steps=30, forecast_steps=15, skip_current_step=True, hidden_size=8, n_attention_heads=2)
        assert model._model is None
        self.__test_fit(
            model,
            endog.copy(),
            exog.copy(),
            batch_size=256,
            epochs=3,
            early_stopping_patience=1,
            validation_split=0.3,
            exog_as_unknown_reals=True,
        )
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15], [e.iloc[-1440 - 45 : -15] for e in exog])
        self.__test_save_and_load(model, TFT, "tft")

    def _test_dlinear(self, endog: pd.DataFrame, exog: list[pd.DataFrame]) -> None:
        model = DLinear(lagged_steps=30, forecast_steps=15, skip_current_step=True, kernel_size=5)
        assert model._model is None
        self.__test_fit(
            model,
            endog.copy(),
            exog.copy(),
            batch_size=256,
            epochs=3,
            early_stopping_patience=1,
            validation_split=0.3,
            exog_as_unknown_reals=True,
        )
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15], [e.iloc[-1440 - 45 : -15] for e in exog])
        self.__test_save_and_load(model, DLinear, "dlinear")

    def _test_patchtst(self, endog: pd.DataFrame, exog: list[pd.DataFrame]) -> None:
        model = PatchTST(
            lagged_steps=30,
            forecast_steps=15,
            skip_current_step=True,
            patch_len=8,
            stride=4,
            d_model=16,
            n_heads=2,
            n_layers=1,
        )
        assert model._model is None
        self.__test_fit(
            model,
            endog.copy(),
            exog.copy(),
            batch_size=256,
            epochs=3,
            early_stopping_patience=1,
            validation_split=0.3,
            exog_as_unknown_reals=True,
        )
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15], [e.iloc[-1440 - 45 : -15] for e in exog])
        self.__test_save_and_load(model, PatchTST, "patchtst")

    def _test_itransformer(self, endog: pd.DataFrame, exog: list[pd.DataFrame]) -> None:
        model = iTransformer(
            lagged_steps=30,
            forecast_steps=15,
            skip_current_step=True,
            d_model=16,
            n_heads=2,
            n_layers=1,
        )
        assert model._model is None
        self.__test_fit(
            model,
            endog.copy(),
            exog.copy(),
            batch_size=256,
            epochs=3,
            early_stopping_patience=1,
            validation_split=0.3,
            exog_as_unknown_reals=True,
        )
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15], [e.iloc[-1440 - 45 : -15] for e in exog])
        self.__test_save_and_load(model, iTransformer, "itransformer")

    def _test_sarima(self, endog: pd.DataFrame) -> None:
        model = SARIMA(lagged_steps=30, forecast_steps=15, skip_current_step=True, ar_order=30, differencing=1)
        assert model._model is None
        self.__test_fit(model, endog.copy(), epochs=3)
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15])
        self.__test_save_and_load(model, SARIMA, "sarima")

    # ------------------------------------------------------------------- shared

    def __test_fit(
        self,
        model: LSTM | DeepAR | SARIMA | TFT | DLinear | PatchTST | iTransformer,
        endog: pd.DataFrame,
        exog: list[pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> None:
        model.fit(endog, exog=exog, **kwargs)
        assert model._model is not None
        if isinstance(model, SARIMA):
            assert model._res is not None
        else:
            assert len(model._endog_scalers) == 2
            if exog is not None:
                assert len(model._exog_scalers) == 2
            assert model._dataset is not None
            if isinstance(exog, list):
                assert all(s > 0 for s in model._exog_size)
            elif exog is not None:
                assert model._exog_size > 0
        assert model._endog_size > 0

    def __test_predict(
        self,
        model: LSTM | DeepAR | SARIMA | TFT | DLinear | PatchTST | iTransformer,
        endog: pd.DataFrame,
        exog: list[pd.DataFrame] | None = None,
    ) -> None:
        preds = model.predict(endog, exog)
        if isinstance(model, SARIMA):
            assert preds.shape[0] == 1
        else:
            # one prediction window per valid encoder anchor
            assert preds.shape[0] == endog.shape[0] - model.lagged_steps
        assert preds.shape[1] == model.forecast_steps
        assert preds.shape[2] == model._endog_size
        assert preds.min() >= 0
        assert preds.max() < np.inf

    def __test_save_and_load(
        self,
        model: LSTM | DeepAR | SARIMA | TFT | DLinear | PatchTST | iTransformer,
        cls: type,
        filename: str,
    ) -> None:
        models_dir = path.join(cur_dir, "models")
        if not path.exists(models_dir):
            makedirs(models_dir)
        filepath = path.join(models_dir, f"{filename}.z")
        model.save(filepath)
        assert path.exists(filepath)

        loaded = cls.load_model(filepath)
        assert loaded._model is not None
        if isinstance(loaded, SARIMA):
            assert loaded._res is not None
        else:
            assert loaded._dataset is not None
