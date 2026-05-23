"""Quantile loss for STARIXNet.

Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory of this source
tree.
"""

from __future__ import annotations

import torch
from torch import nn


class QuantileLoss(nn.Module):
    """Pinball (quantile) loss.

    For a quantile ``q`` and error ``e = y - y_hat``, the loss is
    ``max(q * e, (q - 1) * e)``. With ``q=0.5`` this reduces to half the
    mean absolute error.
    """

    def __init__(self, quantile: float) -> None:
        """Initialize the loss.

        Args:
            quantile: Target quantile in ``(0, 1)``. Values near 0.5 weight over- and under-estimation equally; values
                closer to 0 or 1 penalize one side more heavily.
        """
        super().__init__()
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.quantile = quantile

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the mean quantile loss.

        Args:
            inputs: Predicted values, shape matching ``targets``.
            targets: Observed values.
            sample_weights: Optional per-sample weights broadcastable with ``loss[..., 0]``. If provided, weights are
                multiplied element-wise before reduction.

        Returns:
            Scalar tensor with the mean of the (weighted) quantile loss.
        """
        if inputs.shape != targets.shape:
            raise ValueError(f"shape mismatch: inputs {tuple(inputs.shape)} vs targets {tuple(targets.shape)}")
        error = targets - inputs
        loss = torch.maximum(self.quantile * error, (self.quantile - 1.0) * error)
        if sample_weights is not None:
            if loss.shape[:-1] != sample_weights.shape[:-1]:
                raise ValueError(
                    "sample_weights must match loss shape on all but the last dim; "
                    f"got loss {tuple(loss.shape)}, weights {tuple(sample_weights.shape)}"
                )
            loss = loss * sample_weights
        return loss.mean()
