#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

from collections.abc import Iterable
from os import path, makedirs
from typing import List, Union

import numpy as np
import pandas as pd

from starixnet.models import STARIXNet

cur_dir = path.dirname(path.realpath(__file__))


class TestSTARIXNet(object):
    def tests(self):
        endog = pd.read_csv(path.join(cur_dir, "data", "endog.csv"), index_col=[0]).interpolate()
        exog_1 = pd.read_csv(path.join(cur_dir, "data", "exog_1.csv"), index_col=[0]).interpolate()
        exog_2 = pd.read_csv(path.join(cur_dir, "data", "exog_2.csv"), index_col=[0]).interpolate()
        model = self._init_model()
        assert not model._params  # test that there is no model built
        assert not model._moduls
        self.__test_fit(model, endog, [exog_1, exog_2], loss_q=0.95, sample_weights=False)
        self.__test_predict(model, endog.iloc[-1440-45:-15], [exog_1.iloc[-1440-45:-15], exog_2.iloc[-1440-45:-15]])
        model = self._init_model(fit_multivariate=False)
        self.__test_fit(model, endog, exog_1, loss_q=(0.5, 0.5, 0.95, 0.95), sample_weights=True)
        self.__test_predict(model, endog.iloc[-1440-45:-15], exog_1.iloc[-1440-45:-15])
        self.__test_save(model)
        self.__test_load_model()

    def _init_model(self, fit_multivariate: bool = True):
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
        )

    def __test_fit(
        self,
        model: STARIXNet,
        endog: pd.DataFrame,
        exog: Union[pd.DataFrame, List[pd.DataFrame]],
        loss_q: Union[float, Iterable[float]] = 0.5,
        sample_weights: bool = False,
    ):
        if sample_weights:
            sample_weights = pd.DataFrame(np.random.rand(endog.shape[0]), index=endog.index, columns=['sample_weights'])
        else:
            sample_weights = None
        model.fit(
            endog,
            exog=exog,
            loss_q=loss_q,
            batch_size=256,
            epochs=3,
            early_stopping_patience=1,
            validation_split=0.3,
            sample_weights=sample_weights,
            inspect_data_quality=True,
        )
        assert len(model._endog_scalers) == 2 & len(model._exog_scalers) == 2  # test if scalers are extracted
        assert model._params and model._moduls  # test if parameters were set
        assert (
            model._endog_size > 0
        ) and (
            all([exog_size > 0 for exog_size in model._exog_size]) if isinstance(exog, list) else model._exog_size > 0
        )
        assert model._freqs and 2 * np.pi / 10080 in model._freqs and 2 * np.pi / 1440 in model._freqs
        assert len(model._forward_logic) == 3 + len(exog) if isinstance(exog, list) else 4

    def __test_predict(self, model: STARIXNet, endog: pd.DataFrame, exog: Union[pd.DataFrame, List[pd.DataFrame]]):
        preds = model.predict(endog, exog, aggregated=False)
        assert len(preds) == 3 + len(exog) if isinstance(exog, list) else 4
        assert all([pred.shape[0] == 1440 for pred in preds])
        assert all([pred.shape[-1] == model._endog_size for pred in preds])
        preds = model.predict(endog, exog, aggregated=True)
        assert preds.min() >= 0 and preds.max() < np.inf
        assert preds.shape[0] == 1440 and preds.shape[1] == model.forecast_steps and preds.shape[2] == model._endog_size

    def __test_save(self, model):
        if not path.exists(path.join(cur_dir, "models")):
            makedirs(path.join(cur_dir, "models"))
        model.save(path.join(cur_dir, "models", "starixnet.z"))
        assert path.exists(path.join(cur_dir, "models", "starixnet.z"))

    def __test_load_model(self):
        model = STARIXNet.load_model(path.join(cur_dir, "models", "starixnet.z"))
        assert model._params and model._moduls
        # TODO: should ideally compare every attribute of loaded model with trained model
