#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional

import torch
import torch.nn as nn


class QLoss(nn.modules.loss._Loss):
    def __init__(self, quantile: float):
        """
        Quantile loss.

        A `quantile=0.5` will give half of the mean absolute error as it is calculated as
        `max(squantile * error, (quantile - 1) * error)`.

        Args:
            quantile (float): Quantile value for computing quantile loss. max(q * (y-y_pred), (1-q) * (y_pred-y))
        """
        super(QLoss, self).__init__()
        self.quantile = quantile

    def forward(
            self, inputs: torch.Tensor, targets: torch.Tensor, sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the quantile loss.

        Args:
            inputs (torch.Tensor): Prediction values (i.e. y_pred).
            targets (torch.Tensor): Observed values (i.e. y).
            sample_weights (Optional[torch.Tensor], optional): Weights for obtaining a sample-biased loss calculation.
                Defaults to None.

        Returns:
            torch.Tensor: Mean of the quantile loss.
        """
        assert inputs.shape == targets.shape, "Shapes of predictions and targets must match"
        error = targets - inputs
        loss = torch.max(self.quantile * error, (self.quantile - 1) * error)
        if sample_weights is not None:
            assert loss.shape[:-1] == sample_weights.shape[:-1], "Shapes of predictions and sample weights must match"
            loss = loss * sample_weights
        return torch.mean(loss)
