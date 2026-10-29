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

from benchmark.models import LSTM, SARIMA, TFT, CrossLinear, DeepAR, SeasonalNaive, TimeXer, iTransformer

cur_dir = path.dirname(path.realpath(__file__))


class TestBenchMark:
    def tests(self) -> None:
        endog = pd.read_csv(path.join(cur_dir, "data", "endog.csv"), index_col=[0]).interpolate()
        exog_1 = pd.read_csv(path.join(cur_dir, "data", "exog_1.csv"), index_col=[0]).interpolate()
        exog_2 = pd.read_csv(path.join(cur_dir, "data", "exog_2.csv"), index_col=[0]).interpolate()
        self._test_lstm(endog, [exog_1, exog_2])
        self._test_deepar(endog, [exog_1, exog_2])
        self._test_deepar_exog_ignored_by_default(endog, [exog_1, exog_2])
        self._test_tft(endog, [exog_1, exog_2])
        self._test_crosslinear(endog, [exog_1, exog_2])
        self._test_timexer(endog, [exog_1, exog_2])
        self._test_itransformer(endog, [exog_1, exog_2])
        self._test_sarima(endog, [exog_1, exog_2])
        self._test_seasonal_naive(endog, [exog_1, exog_2])

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
        # exog is past-only: the encoder reads it, the rollout decoder is endog-only
        assert model._model.lstm.input_size > model._endog_size
        assert model._model.decoder_lstm.input_size == model._endog_size
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15], [e.iloc[-1440 - 45 : -15] for e in exog])
        self.__test_save_and_load(model, DeepAR, "deepar")

    def _test_deepar_exog_ignored_by_default(self, endog: pd.DataFrame, exog: list[pd.DataFrame]) -> None:
        # without exog_as_unknown_reals the exog frames are validated and scaled but never reach the network, matching
        # the pre-refactor pytorch-forecasting behaviour
        model = DeepAR(lagged_steps=30, forecast_steps=15, skip_current_step=True, hidden_size=8)
        self.__test_fit(model, endog.copy(), exog.copy(), batch_size=256, epochs=1)
        assert model._model.decoder_lstm is None
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15], [e.iloc[-1440 - 45 : -15] for e in exog])

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

    def _test_crosslinear(self, endog: pd.DataFrame, exog: list[pd.DataFrame]) -> None:
        model = CrossLinear(lagged_steps=30, forecast_steps=15, skip_current_step=True, d_model=16, d_ff=32)
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
        self.__test_save_and_load(model, CrossLinear, "crosslinear")

    def _test_timexer(self, endog: pd.DataFrame, exog: list[pd.DataFrame]) -> None:
        model = TimeXer(
            lagged_steps=30,
            forecast_steps=15,
            skip_current_step=True,
            patch_len=8,
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
        self.__test_save_and_load(model, TimeXer, "timexer")

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

    def _test_sarima(self, endog: pd.DataFrame, exog: list[pd.DataFrame]) -> None:
        model = SARIMA(lagged_steps=30, forecast_steps=15, skip_current_step=True, ar_order=30, differencing=1)
        assert model._model is None
        # exog is accepted and silently dropped, as before the refactor
        self.__test_fit(model, endog.copy(), exog.copy(), epochs=3)
        assert model._exog_size == 0
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15], [e.iloc[-1440 - 45 : -15] for e in exog])
        self.__test_save_and_load(model, SARIMA, "sarima")

    def _test_seasonal_naive(self, endog: pd.DataFrame, exog: list[pd.DataFrame]) -> None:
        auto = SeasonalNaive(lagged_steps=30, forecast_steps=15, skip_current_step=True)
        auto.fit(endog.copy(), [e.copy() for e in exog])
        assert auto._season_length > 0

        n_pred, lagged, season = 1440, 30, 1440
        n_fit = endog.shape[0] - n_pred - 2 * lagged  # leaves room for two follow-up windows
        model = SeasonalNaive(lagged_steps=lagged, forecast_steps=15, skip_current_step=True, enforced_periods=[season])
        # exog is accepted and silently dropped, as with SARIMA
        model.fit(endog.iloc[:n_fit].copy(), [e.iloc[:n_fit].copy() for e in exog])
        assert model._season_length == season
        assert model._exog_size == 0

        window = endog.iloc[n_fit : n_fit + n_pred]
        preds = model.predict(window, [e.iloc[n_fit : n_fit + n_pred] for e in exog])
        assert preds.shape == (n_pred - lagged + 1, 15, model._endog_size)
        # forecast step h of the window ending at global row a replays row a + offset - season
        values = endog[model._endog_labels].to_numpy()
        offsets = np.arange(1, 16) + 1  # skip_current_step
        for i in (0, preds.shape[0] // 2, preds.shape[0] - 1):
            anchor = n_fit + lagged - 1 + i
            np.testing.assert_array_equal(preds[i], values[anchor + offsets - season])

        # stateful: a short follow-up window still looks back a full season through the stored history
        follow = endog.iloc[n_fit + n_pred : n_fit + n_pred + lagged]
        short = model.predict(follow)
        assert short.shape == (1, 15, model._endog_size)
        np.testing.assert_array_equal(short[0], values[n_fit + n_pred + lagged - 1 + offsets - season])

        models_dir = path.join(cur_dir, "models")
        if not path.exists(models_dir):
            makedirs(models_dir)
        filepath = path.join(models_dir, "seasonal_naive.z")
        model.save(filepath)
        loaded = SeasonalNaive.load_model(filepath)
        nxt = endog.iloc[n_fit + n_pred + lagged : n_fit + n_pred + 2 * lagged]
        np.testing.assert_array_equal(loaded.predict(nxt), model.predict(nxt))

        # a window with no season of history before it cannot be forecast
        fresh = SeasonalNaive(lagged_steps=lagged, forecast_steps=15, enforced_periods=[season])
        fresh.fit(endog.iloc[season:n_fit].copy())
        try:
            fresh.predict(endog.iloc[:lagged])
        except ValueError:
            pass
        else:
            raise AssertionError("predict without a season of history must raise")

    # ------------------------------------------------------------------- shared

    def __test_fit(
        self,
        model: LSTM | DeepAR | SARIMA | TFT | CrossLinear | TimeXer | iTransformer,
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
        model: LSTM | DeepAR | SARIMA | TFT | CrossLinear | TimeXer | iTransformer,
        endog: pd.DataFrame,
        exog: list[pd.DataFrame] | None = None,
    ) -> None:
        preds = model.predict(endog, exog)
        if isinstance(model, SARIMA):
            assert preds.shape[0] == 1
        else:
            # pre-refactor (pytorch-forecasting) contract: one forecast per full encoder window, the last of which ends
            # at the newest observation
            assert preds.shape[0] == endog.shape[0] - model.lagged_steps + 1
            self.__test_inference_contract(model, endog, exog, preds)
        assert preds.shape[1] == model.forecast_steps
        assert preds.shape[2] == model._endog_size
        assert preds.min() >= 0
        assert preds.max() < np.inf

    def __test_inference_contract(
        self,
        model: LSTM | DeepAR | TFT | CrossLinear | TimeXer | iTransformer,
        endog: pd.DataFrame,
        exog: list[pd.DataFrame] | None,
        preds: np.ndarray,
    ) -> None:
        scaled_endog, scaled_exog = model._preprocess_transform(
            model._select_endog_columns(endog), model._select_exog_columns([e.copy() for e in exog] if exog else None)
        )
        stacked_exog = model._stack_exog(scaled_exog)
        frame = scaled_endog
        if model._exog_as_unknown_reals and stacked_exog is not None:
            frame = pd.concat([scaled_endog, stacked_exog], axis=1)
        values = frame.to_numpy(dtype=np.float32)

        windows = model._build_predict_windows(scaled_endog, stacked_exog)
        np.testing.assert_array_equal(windows[0], values[: model.lagged_steps])
        np.testing.assert_array_equal(windows[-1], values[-model.lagged_steps :])

        # predict-time windows must be laid out exactly like the training encoder windows
        train_encoder, _ = model._build_windows(scaled_endog, stacked_exog)
        np.testing.assert_array_equal(windows[: train_encoder.shape[0]], train_encoder)

        if exog is not None:
            # time-reversed values: an affine perturbation would be cancelled by models that instance-normalise inputs
            perturbed = [e.iloc[::-1].set_axis(e.index) for e in exog]
            perturbed_preds = model.predict(endog, perturbed)
            if model._exog_as_unknown_reals:
                # every deep-learning benchmark must actually consume exog when asked to
                assert not np.array_equal(perturbed_preds, preds)
            else:
                np.testing.assert_array_equal(perturbed_preds, preds)

    def __test_save_and_load(
        self,
        model: LSTM | DeepAR | SARIMA | TFT | CrossLinear | TimeXer | iTransformer,
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
