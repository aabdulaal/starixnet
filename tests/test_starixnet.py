"""Tests for the STARIXNet model.

Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory of this source
tree.
"""

from __future__ import annotations

from collections.abc import Iterable
from os import makedirs, path

import numpy as np
import pandas as pd

from starixnet.models import STARIXNet

cur_dir = path.dirname(path.realpath(__file__))


class TestSTARIXNet:
    def tests(self) -> None:
        endog = pd.read_csv(path.join(cur_dir, "data", "endog.csv"), index_col=[0]).interpolate()
        exog_1 = pd.read_csv(path.join(cur_dir, "data", "exog_1.csv"), index_col=[0]).interpolate()
        exog_2 = pd.read_csv(path.join(cur_dir, "data", "exog_2.csv"), index_col=[0]).interpolate()
        model = self._init_model()
        assert not model._params
        assert not model._moduls
        self.__test_fit(model, endog, [exog_1, exog_2], loss_q=0.95, sample_weights=False)
        self.__test_predict(
            model,
            endog.iloc[-1440 - 45 : -15],
            [exog_1.iloc[-1440 - 45 : -15], exog_2.iloc[-1440 - 45 : -15]],
        )
        model = self._init_model(fit_multivariate=False)
        self.__test_fit(model, endog, exog_1, loss_q=(0.5, 0.5, 0.95, 0.95), sample_weights=True)
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15], exog_1.iloc[-1440 - 45 : -15])
        model = self._init_model(applied_attributes=["temporal", "autoregressive"])
        self.__test_fit(model, endog, exog_1, loss_q=(0.5, 0.95), sample_weights=False)
        self.__test_predict(model, endog.iloc[-1440 - 45 : -15], exog_1.iloc[-1440 - 45 : -15])
        self.__test_save(model)
        self.__test_load_model()

    def _init_model(
        self,
        fit_multivariate: bool = True,
        applied_attributes: list[str] | None = None,
    ) -> STARIXNet:
        return STARIXNet(
            10080,
            enforced_periods=[10080, 1440],
            max_periods=5,
            min_peak_distance=60,
            min_oscillations=3,
            lagged_steps=30,
            forecast_steps=15,
            skip_current_step=True,
            fit_multivariate=fit_multivariate,
            applied_attributes=applied_attributes,
        )

    def __test_fit(
        self,
        model: STARIXNet,
        endog: pd.DataFrame,
        exog: pd.DataFrame | list[pd.DataFrame],
        loss_q: float | Iterable[float] = 0.5,
        sample_weights: bool = False,
    ) -> None:
        weights = (
            pd.DataFrame(np.random.rand(endog.shape[0]), index=endog.index, columns=["sample_weights"])
            if sample_weights
            else None
        )
        model.fit(
            endog,
            exog=exog,
            loss_q=loss_q,
            batch_size=256,
            epochs=3,
            early_stopping_patience=1,
            validation_split=0.3,
            sample_weights=weights,
            inspect_data_quality=True,
        )
        assert len(model._endog_scalers) == 2
        assert len(model._exog_scalers) == 2
        assert model._forward_logic
        active = set(model._forward_logic)
        if "seasonal" in active:
            assert model._params
        if active & {"temporal", "autoregressive", "exogenous"}:
            assert model._moduls
        assert model._endog_size > 0
        if isinstance(exog, list):
            assert all(size > 0 for size in model._exog_size)
        else:
            assert model._exog_size > 0
        if "seasonal" in model._forward_logic:
            assert model._freqs
            assert 2 * np.pi / 10080 in model._freqs
            assert 2 * np.pi / 1440 in model._freqs

    def __test_predict(
        self,
        model: STARIXNet,
        endog: pd.DataFrame,
        exog: pd.DataFrame | list[pd.DataFrame],
    ) -> None:
        preds = model.predict(endog, exog, aggregated=False)
        assert len(preds) == len(model._decoder_labels)
        assert all(pred.shape[0] == 1440 for pred in preds)
        assert all(pred.shape[-1] == model._endog_size for pred in preds)
        aggregated = model.predict(endog, exog, aggregated=True)
        assert aggregated.min() >= 0
        assert aggregated.max() < np.inf
        assert aggregated.shape == (1440, model.forecast_steps, model._endog_size)

    def __test_save(self, model: STARIXNet) -> None:
        models_dir = path.join(cur_dir, "models")
        if not path.exists(models_dir):
            makedirs(models_dir)
        filepath = path.join(models_dir, "starixnet.z")
        model.save(filepath)
        assert path.exists(filepath)

    def __test_load_model(self) -> None:
        model = STARIXNet.load_model(path.join(cur_dir, "models", "starixnet.z"))
        assert model._forward_logic
        active = set(model._forward_logic)
        if "seasonal" in active:
            assert model._params
        if active & {"temporal", "autoregressive", "exogenous"}:
            assert model._moduls
