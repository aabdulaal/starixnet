"""Benchmark forecasting models implemented directly in PyTorch + Lightning.

Copyright (c) Walmart Inc.

This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory of this source
tree.
"""

from __future__ import annotations

import math
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
from starixnet.utilities import extract_frequencies, make_seq2seq_windows

# -------------------------------------------------------------------------------------------------------------------- #
# Shared encoder/decoder sequence preparation                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


class LightningBaseModel(BaseModel):
    """Base for the deep-learning benchmarks (LSTM, DeepAR, TFT, CrossLinear, TimeXer, iTransformer).

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
        """Sliding-window encoder array for prediction (no decoder targets).

        Returns ``(N - lagged_steps + 1, lagged_steps, channels)``: window ``i`` covers rows ``[i, i + lagged_steps)``
        so the last window ends at the newest observation. This is the pre-refactor ``pytorch-forecasting`` inference
        contract; ``make_seq2seq_windows(return_x_only=True)`` must not be used here because its windows stop one row
        short of their anchor, which silently drops the newest observation and shifts every forecast one step early.
        """
        if endog.shape[0] < self.lagged_steps:
            raise ValueError(f"predict needs at least lagged_steps={self.lagged_steps} rows, got {endog.shape[0]}")
        frame = endog
        if self._exog_as_unknown_reals and exog is not None:
            frame = pd.concat([endog, exog], axis=1)
        values = frame.to_numpy(dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(values, self.lagged_steps, axis=0)
        return np.ascontiguousarray(windows.transpose(0, 2, 1))

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
        # Exog is only observed in the past, so the rollout cannot consume it; a separate endog-only decoder takes
        # over the encoder state. Without exog the encoder LSTM doubles as decoder (keeps the state_dict unchanged).
        self.decoder_lstm = (
            nn.LSTM(
                input_size=endog_size,
                hidden_size=hidden_size,
                num_layers=rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0.0,
                batch_first=True,
            )
            if input_size > endog_size
            else None
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

        Returns ``(mu, sigma)`` each shaped ``(B, future_steps, endog_size)``. Exogenous channels (if any) are only
        read by the encoder; the rollout is fed endogenous values alone.
        """
        _, state = self.lstm(encoder)
        decoder = self.decoder_lstm if self.decoder_lstm is not None else self.lstm
        last_step = encoder[:, -1:, : self.endog_size]

        mus, sigmas = [], []
        for _ in range(future_steps):
            h, state = decoder(last_step, state)
            dist = self._step_distribution(h[:, -1, :])
            mus.append(dist.mean.unsqueeze(1))
            sigmas.append(dist.stddev.unsqueeze(1))
            last_step = dist.mean.unsqueeze(1)
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
# CrossLinear                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


class _CrossLinearNet(pl.LightningModule):
    """CrossLinear (Zhou et al., KDD 2025), following the authors' reference implementation.

    Every channel is instance-normalised over the lookback window. A kernel-3 Conv1d over all channels ([endog | exog])
    produces one cross-correlation series per target, blended with the target's own normalised series through a
    learnable ``alpha``. The blend is split into non-overlapping patches, embedded by an MLP, mixed with a positional
    embedding through a learnable ``beta``, and decoded by an MLP head over the flattened patches; the output is
    de-normalised with the targets' window statistics. The reference ``MS`` mode (one target) is generalised to
    ``endog_size`` targets by widening the Conv1d output, as described in the paper's multivariate extension. Without
    exog the Conv1d mixes the targets only, matching the reference ``M`` mode.
    """

    _EPS = 1e-5

    def __init__(
        self,
        input_size: int,
        endog_size: int,
        lagged_steps: int,
        prediction_length: int,
        patch_len: int,
        d_model: int,
        d_ff: int,
        alpha: float,
        beta: float,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.endog_size = endog_size
        self.lagged_steps = lagged_steps
        self.prediction_length = prediction_length
        self.patch_len = max(1, min(patch_len, lagged_steps))
        self.n_patches = math.ceil(lagged_steps / self.patch_len)
        self.pad_size = self.n_patches * self.patch_len - lagged_steps

        self.alpha = nn.Parameter(torch.ones(1) * alpha)
        self.beta = nn.Parameter(torch.ones(1) * beta)
        self.correlation_embedding = nn.Conv1d(input_size, endog_size, 3, padding="same")
        self.value_embedding = nn.Sequential(
            nn.LayerNorm([endog_size, self.n_patches, self.patch_len]),
            nn.Linear(self.patch_len, d_ff),
            nn.LayerNorm([endog_size, self.n_patches, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm([endog_size, self.n_patches, d_model]),
            nn.ReLU(),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, endog_size, self.n_patches, d_model))
        self.head = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(self.n_patches * d_model, d_ff),
            nn.LayerNorm([endog_size, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, prediction_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) with layout [endog | exog] → (B, C, T)
        x = x.permute(0, 2, 1)
        target = x[:, : self.endog_size, :]
        mean = target.mean(dim=-1, keepdim=True)
        std = target.std(dim=-1, keepdim=True)
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self._EPS)

        h = self.alpha * x[:, : self.endog_size, :] + (1 - self.alpha) * self.correlation_embedding(x)
        h = nn.functional.pad(h, (0, self.pad_size)).unfold(2, self.patch_len, self.patch_len)
        h = self.beta * self.value_embedding(h) + (1 - self.beta) * self.pos_embedding
        out = self.head(h) * std + mean  # (B, E, prediction_length)
        return out.permute(0, 2, 1)

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


class CrossLinear(LightningBaseModel):
    """CrossLinear baseline (Zhou et al., KDD 2025) — linear-family forecaster built for exogenous variables.

    Patch-wise MLP backbone with a lightweight cross-correlation embedding that injects time-invariant dependencies on
    the exogenous signals. Replaces DLinear in the benchmark suite: it keeps the "is STARIXNet's complexity justified?"
    role of the linear family while also consuming the exogenous signals. Defaults follow the authors' ECL scripts.
    """

    def __init__(
        self,
        patch_len: int = 2,
        d_model: int = 256,
        d_ff: int = 1024,
        alpha: float = 1.0,
        beta: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.alpha = alpha
        self.beta = beta

    def _build(self) -> None:
        self._model = _CrossLinearNet(
            input_size=self._total_input_size,
            endog_size=self._endog_size,
            lagged_steps=self.lagged_steps,
            prediction_length=self._prediction_length,
            patch_len=self.patch_len,
            d_model=self.d_model,
            d_ff=self.d_ff,
            alpha=self.alpha,
            beta=self.beta,
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "patch_len": self.patch_len,
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "alpha": self.alpha,
                "beta": self.beta,
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
# TimeXer                                                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #


class _TimeXerBlock(nn.Module):
    """One TimeXer layer: patch-wise self-attention over the endogenous tokens, then cross-attention from the global
    token to the exogenous variate tokens, then a feed-forward pass (post-norm residuals throughout).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, variates: torch.Tensor) -> torch.Tensor:
        # h: (B*E, n_patches + 1, d_model) with the global token last; variates: (B*E, V, d_model)
        attn, _ = self.self_attention(h, h, h, need_weights=False)
        h = self.norm1(h + self.dropout(attn))
        glb = h[:, -1:, :]
        cross, _ = self.cross_attention(glb, variates, variates, need_weights=False)
        glb = self.norm2(glb + self.dropout(cross))
        h = torch.cat([h[:, :-1, :], glb], dim=1)
        return self.norm3(h + self.dropout(self.feed_forward(h)))


class _TimeXerNet(pl.LightningModule):
    """TimeXer (Wang et al., 2024): patch transformer on the targets with cross-attention to exogenous variates.

    Each endogenous series is split into non-overlapping patches plus one learnable global token, and processed with
    shared weights (channel-independent across targets, as in PatchTST). Each exogenous series is embedded whole as a
    single variate token (as in iTransformer) and reaches the target only through the global token's cross-attention.
    Without exog, the targets' own variate tokens are used as the cross-attention context, mirroring the reference
    implementation's multivariate mode.
    """

    def __init__(
        self,
        input_size: int,
        endog_size: int,
        lagged_steps: int,
        prediction_length: int,
        patch_len: int,
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
        # Replicate-pad the tail so lagged_steps is a whole number of patches.
        self.pad_size = -lagged_steps % self.patch_len
        self.n_patches = (lagged_steps + self.pad_size) // self.patch_len

        # d_model must be divisible by n_heads for nn.MultiheadAttention.
        head_dim = max(1, d_model // n_heads)
        self.d_model = head_dim * n_heads

        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
        self.global_token = nn.Parameter(torch.randn(1, endog_size, 1, self.d_model))
        self.variate_embed = nn.Linear(lagged_steps, self.d_model)
        self.layers = nn.ModuleList(_TimeXerBlock(self.d_model, n_heads, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear((self.n_patches + 1) * self.d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) with layout [endog | exog]
        b = x.shape[0]
        endog = x[:, :, : self.endog_size].permute(0, 2, 1)  # (B, E, T)
        if self.pad_size > 0:
            endog = torch.cat([endog, endog[:, :, -1:].repeat(1, 1, self.pad_size)], dim=-1)
        patches = endog.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # (B, E, n_patches, patch_len)
        tokens = self.patch_embed(patches) + self.pos_embed
        glb = self.global_token.expand(b, -1, -1, -1)
        h = torch.cat([tokens, glb], dim=2).reshape(b * self.endog_size, self.n_patches + 1, self.d_model)

        context = x[:, :, self.endog_size :] if x.shape[-1] > self.endog_size else x[:, :, : self.endog_size]
        variates = self.variate_embed(context.permute(0, 2, 1))  # (B, V, d_model)
        variates = variates.repeat_interleave(self.endog_size, dim=0)  # (B*E, V, d_model)

        for layer in self.layers:
            h = layer(h, variates)
        out = self.head(self.dropout(self.norm(h).flatten(start_dim=1)))  # (B*E, prediction_length)
        return out.view(b, self.endog_size, self.prediction_length).permute(0, 2, 1)

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


class TimeXer(LightningBaseModel):
    """TimeXer baseline (Wang et al., 2024) — the exogenous-aware successor of PatchTST.

    Keeps PatchTST's patch-level temporal attention on the targets and adds variate-level cross-attention to the
    exogenous signals through a learnable global token. Replaces PatchTST in the benchmark suite, whose channel-
    independent design cannot consume exog at all.
    """

    def __init__(
        self,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

    def _build(self) -> None:
        self._model = _TimeXerNet(
            input_size=self._total_input_size,
            endog_size=self._endog_size,
            lagged_steps=self.lagged_steps,
            prediction_length=self._prediction_length,
            patch_len=self.patch_len,
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


# -------------------------------------------------------------------------------------------------------------------- #
# Seasonal Naive                                                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


class SeasonalNaive(BaseModel):
    """Seasonal naive forecaster: every future step repeats the value observed one season earlier.

    Training-free sanity baseline (conventionally not counted against the benchmark cap). It answers whether STARIXNet's
    seasonal branch beats simply replaying the last period. The season length is ``max(enforced_periods)`` when given
    (the same period ``BaseModel`` uses for quality checks), else the dominant PSD period found by
    :func:`~starixnet.utilities.extract_frequencies` with STARIXNet's peak-detection defaults. Exogenous variables are
    accepted and silently dropped.

    Like :class:`SARIMA` the model is stateful: it keeps the last season of observations and extends it with every
    ``predict`` input, so callers can keep sending ``lagged_steps``-long windows even when the season is much longer.
    ``predict`` follows the deep benchmarks' contract of one forecast per full ``lagged_steps`` window, the last of
    which ends at the newest observation. Gaps between stored history and new input are forward-filled.
    """

    def __init__(self, min_peak_distance: int = 60, min_oscillations: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_peak_distance = min_peak_distance
        self.min_oscillations = min_oscillations
        self._season_length: int = 0
        self._step_size: int = 0
        self._history: pd.DataFrame | None = None

    # ----------------------------------------------------------------------------------------------------------- public

    def fit(self, endog: pd.DataFrame, exog: ExogFrames | None = None, **kwargs: Any) -> Self:
        endog, _ = super().fit(endog, exog=None, **kwargs)
        self._step_size = int(endog.index[1] - endog.index[0])
        self._season_length = self._resolve_season_length(endog)
        if endog.shape[0] < self._season_length:
            raise ValueError(f"fit needs at least one season ({self._season_length} rows), got {endog.shape[0]}")
        self._history = endog[self._endog_labels].iloc[-self._season_length :].copy()
        return self

    def predict(self, endog: pd.DataFrame, exog: ExogFrames | None = None) -> np.ndarray:
        missing = [f for f in self._endog_labels if f not in endog.columns]
        if missing:
            raise ValueError(f"endog is missing required columns: {missing}")
        if endog.shape[0] < self.lagged_steps:
            raise ValueError(f"predict needs at least lagged_steps={self.lagged_steps} rows, got {endog.shape[0]}")
        endog = endog[self._endog_labels]

        combined = pd.concat([self._history, endog])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        grid = range(int(combined.index[0]), int(combined.index[-1]) + 1, self._step_size)
        combined = combined.reindex(grid).ffill()
        values = combined.to_numpy(dtype=np.float64)

        # anchor of each window (its newest observation) as a position in the regular grid
        anchors = combined.index.get_indexer(endog.index[self.lagged_steps - 1 :])
        offsets = np.arange(1, self.forecast_steps + 1) + int(self.skip_current_step)
        sources = anchors[:, None] + offsets[None, :] - self._season_length * np.ceil(offsets / self._season_length)
        sources = sources.astype(int)
        if sources.min() < 0:
            raise ValueError(
                f"not enough history: the earliest window needs {-sources.min()} more row(s) before it to look back "
                f"one season ({self._season_length} steps); send a longer input or call predict in time order"
            )

        self._history = combined.iloc[-self._season_length :]
        return np.clip(values[sources], 0.0, np.inf)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "min_peak_distance": self.min_peak_distance,
                "min_oscillations": self.min_oscillations,
                "_season_length": self._season_length,
                "_step_size": self._step_size,
                "_history": self._history,
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

    def _resolve_season_length(self, endog: pd.DataFrame) -> int:
        if self.enforced_periods:
            return int(max(self.enforced_periods))
        freqs = extract_frequencies(
            endog,
            distance=self.min_peak_distance,
            min_periods=self.min_oscillations,
            max_freqs=1,
        )
        if not freqs:
            raise ValueError("no seasonal period detected via PSD; pass enforced_periods to set the season length")
        return round(2.0 * np.pi / freqs[0])
