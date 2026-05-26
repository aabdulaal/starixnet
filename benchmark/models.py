"""Benchmark forecasting models implemented directly in PyTorch + Lightning.

Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory of this source
tree.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Self
from zipfile import ZIP_DEFLATED, ZipFile

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from lightning.pytorch.callbacks import EarlyStopping
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from starixnet.models import BaseModel, ExogFrames, _is_multi_exog
from starixnet.utilities import make_seq2seq_windows

# -------------------------------------------------------------------------------------------------------------------- #
# Shared encoder/decoder sequence preparation                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


class LightningBaseModel(BaseModel):
    """Base for the deep-learning benchmarks (LSTM, DeepAR, TFT, DLinear, PatchTST, iTransformer).

    Provides sequence preparation, Lightning training orchestration, and save/load via a zip archive (joblib config +
    ``state_dict``). All subclasses share the same ``fit`` / ``predict`` / ``save`` / ``load_model`` signature so they
    are drop-in interchangeable in an inference service.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # forecast_steps + 1 buffer when skipping current step; the buffer's first prediction is discarded at output.
        self._prediction_length = self.forecast_steps + (1 if self.skip_current_step else 0)
        self._model: pl.LightningModule | None = None
        self._dataset: dict[str, Any] | None = None
        self._exog_as_unknown_reals: bool = False

    # ------------------------------------------------------------------------------------------------------------ utils

    @property
    def _total_input_size(self) -> int:
        """Number of channels in the model's per-step input vector."""
        if self._exog_as_unknown_reals and self._exog_size:
            extra = sum(self._exog_size) if isinstance(self._exog_size, list) else self._exog_size
            return self._endog_size + extra
        return self._endog_size

    def _stack_exog(self, exog: ExogFrames | None) -> pd.DataFrame | None:
        if exog is None:
            return None
        if _is_multi_exog(exog):
            return pd.concat(exog, axis=1)
        return exog

    def _build_windows(
        self,
        endog: pd.DataFrame,
        exog: pd.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(encoder, decoder)`` windowed arrays with shapes
        ``(N, lagged_steps, channels)`` and ``(N, prediction_length, endog_size)``.
        """
        endog_x, endog_y, _ = make_seq2seq_windows(
            endog,
            lagged_steps=self.lagged_steps,
            forecast_steps=self._prediction_length,
            lagged_features=endog.columns.to_list(),
            skip_current_step=False,  # buffer step handled via _prediction_length
        )

        encoder = endog_x.to_numpy(dtype=np.float32).reshape(
            -1,
            self.lagged_steps,
            self._endog_size,
        )
        decoder = endog_y.to_numpy(dtype=np.float32).reshape(
            -1,
            self._prediction_length,
            self._endog_size,
        )

        if self._exog_as_unknown_reals and exog is not None:
            exog_x, _, _ = make_seq2seq_windows(
                exog,
                lagged_steps=self.lagged_steps,
                forecast_steps=self._prediction_length,
                lagged_features=exog.columns.to_list(),
                skip_current_step=False,
            )
            exog_arr = exog_x.to_numpy(dtype=np.float32).reshape(
                -1,
                self.lagged_steps,
                exog.shape[1],
            )
            encoder = np.concatenate([encoder, exog_arr], axis=-1)

        return encoder, decoder

    def _build_predict_windows(
        self,
        endog: pd.DataFrame,
        exog: pd.DataFrame | None,
    ) -> np.ndarray:
        """Sliding-window encoder array for prediction (no decoder targets)."""
        encoder_only = make_seq2seq_windows(
            endog,
            lagged_steps=self.lagged_steps,
            forecast_steps=0,
            lagged_features=endog.columns.to_list(),
            return_x_only=True,
        )
        encoder = encoder_only.to_numpy(dtype=np.float32).reshape(
            -1,
            self.lagged_steps,
            self._endog_size,
        )
        if self._exog_as_unknown_reals and exog is not None:
            exog_only = make_seq2seq_windows(
                exog,
                lagged_steps=self.lagged_steps,
                forecast_steps=0,
                lagged_features=exog.columns.to_list(),
                return_x_only=True,
            )
            exog_arr = exog_only.to_numpy(dtype=np.float32).reshape(
                -1,
                self.lagged_steps,
                exog.shape[1],
            )
            encoder = np.concatenate([encoder, exog_arr], axis=-1)
        return encoder

    # ----------------------------------------------------------------------------------------------------------- public

    def fit(
        self,
        endog: pd.DataFrame,
        exog: ExogFrames | None = None,
        batch_size: int = 32,
        epochs: int = 100,
        shuffle: bool = True,
        early_stopping_patience: int = -1,
        validation_split: float = 0.0,
        sample_weights: pd.DataFrame | None = None,
        exog_as_unknown_reals: bool = False,
        gradient_clip_val: float | None = None,
        **kwargs: Any,
    ) -> Self:
        endog, exog = super().fit(endog, exog=exog, **kwargs)
        self._exog_as_unknown_reals = exog_as_unknown_reals
        endog, exog = self._preprocess_fit(endog, exog)._preprocess_transform(endog, exog)

        exog_stacked = self._stack_exog(exog)
        encoder, decoder = self._build_windows(endog, exog_stacked)

        if validation_split > 0.0:
            cutoff = max(1, round(validation_split * encoder.shape[0]))
            train_enc, val_enc = encoder[:-cutoff], encoder[-cutoff:]
            train_dec, val_dec = decoder[:-cutoff], decoder[-cutoff:]
        else:
            train_enc, val_enc = encoder, None
            train_dec, val_dec = decoder, None

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_enc), torch.from_numpy(train_dec)),
            batch_size=batch_size,
            shuffle=shuffle,
        )
        val_loader = (
            DataLoader(
                TensorDataset(torch.from_numpy(val_enc), torch.from_numpy(val_dec)),
                batch_size=batch_size,
                shuffle=False,
            )
            if val_enc is not None
            else None
        )

        # save a minimal sample for later reconstruction in load_model
        self._dataset = {
            "encoder_sample": encoder[-1:].copy(),
            "decoder_sample": decoder[-1:].copy(),
            "input_size": self._total_input_size,
            "endog_size": self._endog_size,
        }

        if self._model is None:
            self._build()

        callbacks: list[pl.Callback] = []
        if early_stopping_patience > 0 and val_loader is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-4,
                    patience=early_stopping_patience,
                    mode="min",
                )
            )

        trainer = pl.Trainer(
            accelerator="cpu",
            callbacks=callbacks,
            max_epochs=epochs,
            gradient_clip_val=gradient_clip_val,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
        )
        trainer.fit(self._model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        self._assert_no_nan_params()
        return self

    def predict(
        self,
        endog: pd.DataFrame,
        exog: ExogFrames | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        endog = self._select_endog_columns(endog)
        exog = self._select_exog_columns(exog)
        endog, exog = self._preprocess_transform(endog, exog)
        exog_stacked = self._stack_exog(exog)

        encoder = self._build_predict_windows(endog, exog_stacked)
        loader = DataLoader(TensorDataset(torch.from_numpy(encoder)), batch_size=batch_size, shuffle=False)

        self._model.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in loader:
                preds = self._model.predict_step(batch, self._prediction_length)
                chunks.append(preds.cpu().numpy()[:, -self.forecast_steps :, :])
        return self._inverse_transform(np.concatenate(chunks, axis=0))

    def save(self, filepath: str | BytesIO, compress: bool = True) -> None:
        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as zf:
            config_buf = BytesIO()
            dump(self.get_config(), config_buf, compress=compress)
            zf.writestr("config_file.z", config_buf.getvalue())

            state_buf = BytesIO()
            torch.save(self._model.state_dict(), state_buf)
            zf.writestr("model_state.z", state_buf.getvalue())
        dump(zip_buffer.getvalue(), filepath, compress=compress)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "_prediction_length": self._prediction_length,
                "_dataset": self._dataset,
                "_exog_as_unknown_reals": self._exog_as_unknown_reals,
            }
        )
        return config

    @classmethod
    def _load_archive(cls, filepath: str | BytesIO) -> tuple[Self, dict[str, Any]]:
        """Hydrate config and return ``(instance, state_dict)`` for subclasses."""
        instance = cls(0)
        zipped_bytes = load(filepath)
        with ZipFile(BytesIO(zipped_bytes), "r") as zf:
            config = load(BytesIO(zf.read("config_file.z")))
            for param, val in config.items():
                setattr(instance, param, val)
            state_dict = torch.load(BytesIO(zf.read("model_state.z")), weights_only=True)
        return instance, state_dict

    def _select_endog_columns(self, endog: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self._endog_labels if f not in endog.columns]
        if missing:
            raise ValueError(f"endog is missing required columns: {missing}")
        return endog[self._endog_labels]

    def _select_exog_columns(self, exog: ExogFrames | None) -> ExogFrames | None:
        if exog is None:
            return None
        if _is_multi_exog(exog):
            selected: list[pd.DataFrame] = []
            for i, frame in enumerate(exog):
                missing = [f for f in self._exog_labels[i] if f not in frame.columns]
                if missing:
                    raise ValueError(f"exog group {i} is missing required columns: {missing}")
                selected.append(frame[self._exog_labels[i]])
            return selected
        missing = [f for f in self._exog_labels if f not in exog.columns]
        if missing:
            raise ValueError(f"exog is missing required columns: {missing}")
        return exog[self._exog_labels]

    def _assert_no_nan_params(self) -> None:
        for param in self._model.parameters():
            if torch.isnan(param).any():
                raise RuntimeError(
                    "Training corrupted: NaN found in network parameters. Inspect inputs and configuration."
                )

    def _build(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


# -------------------------------------------------------------------------------------------------------------------- #
# LSTM                                                                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #


class _LSTMNet(pl.LightningModule):
    """Encoder LSTM + linear projection over the prediction horizon."""

    def __init__(
        self,
        input_size: int,
        endog_size: int,
        hidden_size: int,
        rnn_layers: int,
        dropout: float,
        prediction_length: int,
    ) -> None:
        super().__init__()
        self.endog_size = endog_size
        self.prediction_length = prediction_length
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, endog_size * prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        flat = self.head(h[-1])
        return flat.view(-1, self.prediction_length, self.endog_size)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())

    def predict_step(self, x: torch.Tensor, _: int) -> torch.Tensor:
        return self(x)


class LSTM(LightningBaseModel):
    """LSTM encoder followed by a single linear projection over the horizon."""

    def __init__(
        self,
        hidden_size: int = 1,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout

    def _build(self) -> None:
        self._model = _LSTMNet(
            input_size=self._total_input_size,
            endog_size=self._endog_size,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            prediction_length=self._prediction_length,
        )

    def get_config(self) -> dict[str, Any]:
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
    def load_model(cls, filepath: str | BytesIO) -> Self:
        instance, state_dict = cls._load_archive(filepath)
        instance._build()
        instance._model.load_state_dict(state_dict)
        return instance


# -------------------------------------------------------------------------------------------------------------------- #
# DeepAR                                                                                                               #
# -------------------------------------------------------------------------------------------------------------------- #


class _DeepARNet(pl.LightningModule):
    """Probabilistic LSTM with per-step Gaussian outputs (DeepAR style)."""

    def __init__(
        self,
        input_size: int,
        endog_size: int,
        hidden_size: int,
        rnn_layers: int,
        dropout: float,
        prediction_length: int,
    ) -> None:
        super().__init__()
        self.endog_size = endog_size
        self.prediction_length = prediction_length
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
        )
        # outputs (mu, log_sigma) per target
        self.mu_head = nn.Linear(hidden_size, endog_size)
        self.log_sigma_head = nn.Linear(hidden_size, endog_size)

    def _step_distribution(self, h: torch.Tensor) -> torch.distributions.Normal:
        mu = self.mu_head(h)
        sigma = torch.nn.functional.softplus(self.log_sigma_head(h)) + 1e-6
        return torch.distributions.Normal(mu, sigma)

    def forward(
        self,
        encoder: torch.Tensor,
        future_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively roll out ``future_steps`` predictions.

        Returns ``(mu, sigma)`` each shaped ``(B, future_steps, endog_size)``. The last observed step's exogenous
        channels (if any) are carried forward unchanged at every rollout step.
        """
        _, state = self.lstm(encoder)
        last_step = encoder[:, -1:, :]  # (B, 1, input_size)
        has_exog = last_step.shape[-1] > self.endog_size
        exog_tail = last_step[:, :, self.endog_size :] if has_exog else None

        mus, sigmas = [], []
        for _ in range(future_steps):
            h, state = self.lstm(last_step, state)
            dist = self._step_distribution(h[:, -1, :])
            mus.append(dist.mean.unsqueeze(1))
            sigmas.append(dist.stddev.unsqueeze(1))
            next_endog = dist.mean.unsqueeze(1)
            last_step = torch.cat([next_endog, exog_tail], dim=-1) if has_exog else next_endog
        return torch.cat(mus, dim=1), torch.cat(sigmas, dim=1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        mu, sigma = self(x, y.shape[1])
        dist = torch.distributions.Normal(mu, sigma)
        loss = -dist.log_prob(y).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        mu, sigma = self(x, y.shape[1])
        dist = torch.distributions.Normal(mu, sigma)
        loss = -dist.log_prob(y).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())

    def predict_step(self, x: torch.Tensor, future_steps: int) -> torch.Tensor:
        mu, _ = self(x, future_steps)
        return mu


class DeepAR(LightningBaseModel):
    """Probabilistic recurrent forecaster with per-step Gaussian outputs.

    Implements the DeepAR family without the original ``pytorch-forecasting`` dependency. Trained by maximum likelihood
    under a Normal observation model and rolled out autoregressively at inference.
    """

    def __init__(
        self,
        hidden_size: int = 1,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout = dropout

    def _build(self) -> None:
        self._model = _DeepARNet(
            input_size=self._total_input_size,
            endog_size=self._endog_size,
            hidden_size=self.hidden_size,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            prediction_length=self._prediction_length,
        )

    def get_config(self) -> dict[str, Any]:
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
    def load_model(cls, filepath: str | BytesIO) -> Self:
        instance, state_dict = cls._load_archive(filepath)
        instance._build()
        instance._model.load_state_dict(state_dict)
        return instance


# Backwards-compatible alias from the previous naming.
DeepVAR = DeepAR


# -------------------------------------------------------------------------------------------------------------------- #
# TFT (pragmatic implementation)                                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


class _TFTNet(pl.LightningModule):
    """Pragmatic Temporal Fusion Transformer-style decoder.

    Implements the key ingredients (LSTM encoder, multi-head self-attention, quantile output, gated residual) without
    the full variable-selection machinery of the original paper. Sufficient as a Transformer baseline that is directly
    comparable to STARIXNet.
    """

    def __init__(
        self,
        input_size: int,
        endog_size: int,
        hidden_size: int,
        lstm_layers: int,
        dropout: float,
        quantiles: list[float],
        prediction_length: int,
        n_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.endog_size = endog_size
        self.prediction_length = prediction_length
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        # ensure hidden dim divisible by heads
        head_dim = max(1, hidden_size // n_attention_heads)
        self.hidden_size = head_dim * n_attention_heads

        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.norm2 = nn.LayerNorm(self.hidden_size)
        # produces quantile outputs per target over the full horizon
        self.head = nn.Linear(self.hidden_size, endog_size * prediction_length * self.n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        attn_out, _ = self.attention(h, h, h, need_weights=False)
        h = self.norm(h + attn_out)
        h = self.norm2(h + self.feed_forward(h))
        # take the last time step as the decoder context
        out = self.head(h[:, -1, :])
        return out.view(-1, self.prediction_length, self.endog_size, self.n_quantiles)

    def _quantile_loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds: (B, T, C, Q); target: (B, T, C)
        quantiles = torch.tensor(self.quantiles, device=preds.device, dtype=preds.dtype)
        error = target.unsqueeze(-1) - preds
        loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
        return loss.mean()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = self._quantile_loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = self._quantile_loss(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())

    def predict_step(self, x: torch.Tensor, _: int) -> torch.Tensor:
        preds = self(x)
        # collapse quantile dim with the median (or central quantile)
        mid = len(self.quantiles) // 2
        return preds[..., mid]


class TFT(LightningBaseModel):
    """Temporal Fusion Transformer baseline (pragmatic reimplementation).

    Captures the key TFT components — LSTM encoder, multi-head self-attention, gated residual, and quantile output —
    without the original variable selection networks. See :class:`_TFTNet` for the architecture.
    """

    def __init__(
        self,
        hidden_size: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        quantiles: list[float] | None = None,
        n_attention_heads: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.quantiles = list(quantiles) if quantiles else [0.5]
        self.n_attention_heads = n_attention_heads
        self.output_size: int | list[int] = len(self.quantiles)

    def _build(self) -> None:
        self._model = _TFTNet(
            input_size=self._total_input_size,
            endog_size=self._endog_size,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            quantiles=self.quantiles,
            prediction_length=self._prediction_length,
            n_attention_heads=self.n_attention_heads,
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "lstm_layers": self.lstm_layers,
                "dropout": self.dropout,
                "quantiles": self.quantiles,
                "n_attention_heads": self.n_attention_heads,
                "output_size": self.output_size,
            }
        )
        return config

    @classmethod
    def load_model(cls, filepath: str | BytesIO) -> Self:
        instance, state_dict = cls._load_archive(filepath)
        instance._build()
        instance._model.load_state_dict(state_dict)
        return instance


# -------------------------------------------------------------------------------------------------------------------- #
# DLinear                                                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #


class _SeriesDecomp(nn.Module):
    """Trend / seasonal split via a centred moving average (Zeng et al., 2022).

    Pads the input with the boundary values so the trend output matches the input length. The seasonal component is the
    residual.
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C). AvgPool1d expects (B, C, T), so we permute around the pool call.
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, 0:1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        padded = torch.cat([front, x, end], dim=1).permute(0, 2, 1)
        trend = self.avg(padded).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend


class _DLinearNet(pl.LightningModule):
    """Decomposition-Linear forecaster (Zeng et al., 2022).

    Splits the encoder window into trend + seasonal via moving-average decomposition and applies a dedicated linear
    projection over the time dimension to each component. Channel-independent in the ``individual=True`` mode.
    """

    def __init__(
        self,
        input_size: int,
        endog_size: int,
        lagged_steps: int,
        prediction_length: int,
        kernel_size: int,
        individual: bool,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.endog_size = endog_size
        self.lagged_steps = lagged_steps
        self.prediction_length = prediction_length
        self.individual = individual
        self.decomp = _SeriesDecomp(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(lagged_steps, prediction_length) for _ in range(input_size)]
            )
            self.linear_trend = nn.ModuleList([nn.Linear(lagged_steps, prediction_length) for _ in range(input_size)])
        else:
            self.linear_seasonal = nn.Linear(lagged_steps, prediction_length)
            self.linear_trend = nn.Linear(lagged_steps, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        seasonal, trend = self.decomp(x)
        seasonal = seasonal.permute(0, 2, 1)  # (B, C, T)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.stack(
                [self.linear_seasonal[i](seasonal[:, i, :]) for i in range(self.input_size)],
                dim=1,
            )
            trend_out = torch.stack(
                [self.linear_trend[i](trend[:, i, :]) for i in range(self.input_size)],
                dim=1,
            )
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)

        out = (seasonal_out + trend_out).permute(0, 2, 1)  # (B, prediction_length, C)
        # Encoder layout is [endog | exog]; keep only the endog channels in the forecast.
        return out[:, :, : self.endog_size]

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())

    def predict_step(self, x: torch.Tensor, _: int) -> torch.Tensor:
        return self(x)


class DLinear(LightningBaseModel):
    """Decomposition-Linear baseline (Zeng et al., 2022).

    A minimal-parameter forecaster that splits the encoder window into trend + seasonal via moving average and applies
    independent linear projections over the time dimension. Serves as the architectural sanity check for whether more
    complex models in the benchmark are justified.
    """

    def __init__(
        self,
        kernel_size: int = 25,
        individual: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.individual = individual

    def _build(self) -> None:
        self._model = _DLinearNet(
            input_size=self._total_input_size,
            endog_size=self._endog_size,
            lagged_steps=self.lagged_steps,
            prediction_length=self._prediction_length,
            kernel_size=self.kernel_size,
            individual=self.individual,
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "individual": self.individual,
            }
        )
        return config

    @classmethod
    def load_model(cls, filepath: str | BytesIO) -> Self:
        instance, state_dict = cls._load_archive(filepath)
        instance._build()
        instance._model.load_state_dict(state_dict)
        return instance


# -------------------------------------------------------------------------------------------------------------------- #
# PatchTST                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #


class _PatchTSTNet(pl.LightningModule):
    """Channel-independent patch transformer (Nie et al., 2023).

    Each channel's lagged window is split into overlapping patches and passed through a shared transformer encoder. The
    head projects the flattened patch embeddings to the forecast horizon. Channel-independence means cross-variate
    dependencies are not modelled here — that role is reserved for iTransformer in the benchmark suite.
    """

    def __init__(
        self,
        input_size: int,
        endog_size: int,
        lagged_steps: int,
        prediction_length: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.endog_size = endog_size
        self.lagged_steps = lagged_steps
        self.prediction_length = prediction_length
        self.patch_len = max(1, min(patch_len, lagged_steps))
        self.stride = max(1, min(stride, self.patch_len))

        # Replicate-pad the tail so (lagged_steps - patch_len) is divisible by stride and the patch count is integer.
        excess = (lagged_steps - self.patch_len) % self.stride
        self.pad_size = (self.stride - excess) if excess > 0 else 0
        padded_len = lagged_steps + self.pad_size
        self.n_patches = (padded_len - self.patch_len) // self.stride + 1

        # d_model must be divisible by n_heads for nn.MultiheadAttention.
        head_dim = max(1, d_model // n_heads)
        self.d_model = head_dim * n_heads

        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * self.d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T) for time-axis patching.
        b, _, c = x.shape
        x = x.permute(0, 2, 1)
        if self.pad_size > 0:
            tail = x[:, :, -1:].repeat(1, 1, self.pad_size)
            x = torch.cat([x, tail], dim=-1)
        # Unfold the time dim into overlapping patches: (B, C, n_patches, patch_len).
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Channel-independent: collapse (B, C) into one batch dim so all channels share the same transformer.
        patches = patches.reshape(b * c, self.n_patches, self.patch_len)

        h = self.patch_embed(patches) + self.pos_embed
        h = self.encoder(h)
        h = h.flatten(start_dim=1)  # (B*C, n_patches * d_model)
        out = self.head(h)  # (B*C, prediction_length)
        out = out.reshape(b, c, self.prediction_length).permute(0, 2, 1)  # (B, prediction_length, C)
        # Encoder layout is [endog | exog]; keep only the endog channels in the forecast.
        return out[:, :, : self.endog_size]

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())

    def predict_step(self, x: torch.Tensor, _: int) -> torch.Tensor:
        return self(x)


class PatchTST(LightningBaseModel):
    """Channel-independent patch-based transformer (Nie et al., 2023).

    Splits each channel's lagged window into overlapping patches, processes them with a shared transformer encoder,
    and projects the flattened patch embeddings to the forecast horizon. Channel-independence makes the model robust
    to distribution shift in unrelated channels at the cost of ignoring cross-variate dependencies.
    """

    def __init__(
        self,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

    def _build(self) -> None:
        self._model = _PatchTSTNet(
            input_size=self._total_input_size,
            endog_size=self._endog_size,
            lagged_steps=self.lagged_steps,
            prediction_length=self._prediction_length,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "patch_len": self.patch_len,
                "stride": self.stride,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def load_model(cls, filepath: str | BytesIO) -> Self:
        instance, state_dict = cls._load_archive(filepath)
        instance._build()
        instance._model.load_state_dict(state_dict)
        return instance


# -------------------------------------------------------------------------------------------------------------------- #
# iTransformer                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


class _iTransformerNet(pl.LightningModule):
    """Inverted Transformer (Liu et al., 2024) — attention over variates rather than over time.

    Embeds each variate's full lagged window as a single token, then applies multi-head self-attention across the
    variate dimension. Output projection maps the resulting per-variate embedding back to the forecast horizon.
    """

    def __init__(
        self,
        input_size: int,
        endog_size: int,
        lagged_steps: int,
        prediction_length: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.endog_size = endog_size
        self.lagged_steps = lagged_steps
        self.prediction_length = prediction_length

        # d_model must be divisible by n_heads for nn.MultiheadAttention.
        head_dim = max(1, d_model // n_heads)
        self.d_model = head_dim * n_heads

        self.token_embed = nn.Linear(lagged_steps, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.projection = nn.Linear(self.d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T) so each variate becomes a token-length sequence.
        x = x.permute(0, 2, 1)
        h = self.token_embed(x)  # (B, C, d_model)
        h = self.encoder(h)
        out = self.projection(h)  # (B, C, prediction_length)
        out = out.permute(0, 2, 1)  # (B, prediction_length, C)
        # Encoder layout is [endog | exog]; keep only the endog channels in the forecast.
        return out[:, :, : self.endog_size]

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())

    def predict_step(self, x: torch.Tensor, _: int) -> torch.Tensor:
        return self(x)


class iTransformer(LightningBaseModel):
    """Inverted Transformer baseline (Liu et al., 2024).

    Treats each variate as a token and applies attention across the variate dimension. Most directly relevant to the
    multivariate framing of STARIXNet — answers whether modelling cross-variate dependencies via attention beats the
    per-variate Conv1d pooling used in STARIXNet's autoregressive and exogenous decoders.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

    def _build(self) -> None:
        self._model = _iTransformerNet(
            input_size=self._total_input_size,
            endog_size=self._endog_size,
            lagged_steps=self.lagged_steps,
            prediction_length=self._prediction_length,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def load_model(cls, filepath: str | BytesIO) -> Self:
        instance, state_dict = cls._load_archive(filepath)
        instance._build()
        instance._model.load_state_dict(state_dict)
        return instance


# -------------------------------------------------------------------------------------------------------------------- #
# SARIMA                                                                                                               #
# -------------------------------------------------------------------------------------------------------------------- #


class SARIMA(BaseModel):
    """Seasonal ARIMA wrapper around :class:`statsmodels.tsa.statespace.SARIMAX`.

    Univariate models are fit per target series. Exogenous variables are currently ignored (a future enhancement is to
    support them as external regressors).
    """

    def __init__(
        self,
        ar_order: int = 1,
        differencing: int = 1,
        ma_order: int = 0,
        seasonal_ar_order: int = 0,
        seasonal_differencing: int = 0,
        seasonal_ma_order: int = 0,
        season_length: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.order: tuple[int, int, int] = (ar_order, differencing, ma_order)
        self.seasonal_order: tuple[int, int, int, int] = (
            seasonal_ar_order,
            seasonal_differencing,
            seasonal_ma_order,
            season_length,
        )

        self._model: SARIMAX | list[SARIMAX] | None = None
        self._res: SARIMAXResults | list[SARIMAXResults] | None = None
        self._state_timestamp: int = 0
        self._state_size: int = 0
        self._step_size: int = 0
        self._prediction_length = self.forecast_steps + (1 if self.skip_current_step else 0)

    # ------------------------------------------------------------------------------------------------------------ build

    def _build(self, endog: pd.DataFrame) -> Self:
        kwargs = {"order": self.order, "seasonal_order": self.seasonal_order, "simple_differencing": False}
        if self._endog_size > 1:
            self._model = [SARIMAX(endog[label].values.reshape(-1), **kwargs) for label in self._endog_labels]
        else:
            self._model = SARIMAX(endog.values.reshape(-1), **kwargs)
        return self

    # ----------------------------------------------------------------------------------------------------------- public

    def fit(
        self,
        endog: pd.DataFrame,
        exog: ExogFrames | None = None,
        epochs: int = 100,
        low_memory: bool = True,
        max_state_size: int | None = None,
        **kwargs: Any,
    ) -> Self:
        endog, _ = super().fit(endog, exog=None, **kwargs)
        self._state_timestamp = int(endog.index[-1])
        endog_size = endog.shape[0]
        self._state_size = min(max_state_size, endog_size) if max_state_size else endog_size
        self._step_size = int(endog.index[1] - endog.index[0])

        if self._model is None:
            self._build(endog)

        if self._endog_size > 1:
            self._res = [
                self._fit_one(model, epochs=epochs, low_memory=low_memory, trim=max_state_size is not None)
                for model in self._model
            ]
        else:
            self._res = self._fit_one(
                self._model,
                epochs=epochs,
                low_memory=low_memory,
                trim=max_state_size is not None,
            )
        return self

    def predict(
        self,
        endog: pd.DataFrame,
        exog: ExogFrames | None = None,
    ) -> np.ndarray:
        missing_state_size = int(((endog.index[0] - self._state_timestamp) // self._step_size) - 1)
        if self._endog_size > 1:
            self._res = [
                self._update_state(res, missing_state_size, endog[label])
                for res, label in zip(self._res, self._endog_labels, strict=True)
            ]
            forecasts = np.concatenate([self._forecast(res) for res in self._res], axis=-1)
        else:
            self._res = self._update_state(self._res, missing_state_size, endog)
            forecasts = self._forecast(self._res)
        if int(endog.index[-1]) > self._state_timestamp:
            self._state_timestamp = int(endog.index[-1])
        return np.clip(forecasts, 0.0, np.inf)

    def get_config(self) -> dict[str, Any]:
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

    @classmethod
    def load_model(cls, filepath: str | BytesIO) -> Self:
        instance = cls(0)
        config = load(filepath)
        for param, val in config.items():
            setattr(instance, param, val)
        return instance

    # ---------------------------------------------------------------------------------------------------------- private

    _RES_ARRAYS_TRAILING_KEEP = (
        # attributes indexed as ``[-(state_size - 1):]`` along the last axis
        "filtered_state",
        "predicted_state",  # special-cased: keeps full state_size
        "forecasts",
        "forecasts_error",
        "standardized_forecasts_error",
        "scaled_smoothed_estimator",
        "smoothing_error",
        "smoothed_state",
        "smoothed_measurement_disturbance",
        "smoothed_state_disturbance",
    )

    def _fit_one(
        self,
        model: SARIMAX,
        epochs: int,
        low_memory: bool,
        trim: bool,
    ) -> SARIMAXResults:
        res = model.fit(maxiter=epochs, low_memory=low_memory)
        if not trim:
            return res

        keep = self._state_size
        keep_minus_one = max(0, keep - 1)

        res.data.endog = res.data.endog[-keep:]
        res.data.data.orig_endog = res.data.data.orig_endog[-keep:]

        for attr in ("filtered", "filtered_cov", "predicted", "predicted_cov", "smoothed", "smoothed_cov"):
            setattr(res.states, attr, getattr(res.states, attr)[-keep:])

        # 2-d / 3-d state arrays
        res.filtered_state = res.filtered_state[:, -keep_minus_one:]
        res.filtered_state_cov = res.filtered_state_cov[:, :, -keep_minus_one:]
        res.predicted_state = res.predicted_state[:, -keep:]
        res.predicted_state_cov = res.predicted_state_cov[:, :, -keep:]

        res.forecasts = res.forecasts[:, -keep_minus_one:]
        res.forecasts_error = res.forecasts_error[:, -keep_minus_one:]
        res.forecasts_error_cov = res.forecasts_error_cov[:, :, -keep_minus_one:]
        res.standardized_forecasts_error = res.standardized_forecasts_error[:, -keep_minus_one:]

        res.scaled_smoothed_estimator = res.scaled_smoothed_estimator[:, -keep_minus_one:]
        res.scaled_smoothed_estimator_cov = res.scaled_smoothed_estimator_cov[:, :, -keep_minus_one:]
        res.smoothing_error = res.smoothing_error[:, -keep_minus_one:]
        res.smoothed_state = res.smoothed_state[:, -keep_minus_one:]
        res.smoothed_state_cov = res.smoothed_state_cov[:, :, -keep_minus_one:]
        res.smoothed_state_autocov = res.smoothed_state_autocov[:, :, -keep_minus_one:]
        res.smoothed_measurement_disturbance = res.smoothed_measurement_disturbance[:, -keep_minus_one:]
        res.smoothed_state_disturbance = res.smoothed_state_disturbance[:, -keep_minus_one:]
        res.smoothed_measurement_disturbance_cov = res.smoothed_measurement_disturbance_cov[:, :, -keep_minus_one:]
        res.smoothed_state_disturbance_cov = res.smoothed_state_disturbance_cov[:, :, -keep_minus_one:]
        return res

    def _update_state(
        self,
        res: SARIMAXResults,
        missing_state_size: int,
        endog: pd.Series | pd.DataFrame,
    ) -> SARIMAXResults:
        if missing_state_size > 0:
            res = res.append(res.forecast(missing_state_size))
        elif missing_state_size < 0:
            endog = endog.iloc[-missing_state_size:]
        if endog.shape[0] > 0:
            res = res.append(endog.values.reshape(-1))
        res.data.endog = res.data.endog[-self._state_size :]
        return res

    def _forecast(self, res: SARIMAXResults) -> np.ndarray:
        return res.forecast(self._prediction_length)[-self.forecast_steps :].reshape(1, -1, 1)
