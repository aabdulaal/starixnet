# STARIXNet

Encoder–decoder neural network for multivariate, multi-attribute time-series forecasting. The model decomposes a forecast into independent decoder branches — **S**easonal, **T**emporal, **A**uto**R**egressive **I**ntegrated, and e**X**ogenous — then aggregates them via a quantile reduction at prediction time. Benchmarks against LSTM, DeepAR, TFT, and SARIMA are included.

## Repository layout

```
starixnet/      # the proposed STARIXNet model (models.py), training loop (utilities.py), and quantile loss (losses.py)
benchmark/      # LSTM, DeepAR, TFT, and SARIMA implemented directly in PyTorch + Lightning
tests/          # unit tests + cached test data and serialized models
```

## Requirements

- Python ≥ 3.11
- Either [`uv`](https://docs.astral.sh/uv/) (recommended) or plain `pip`

## Installation

### Option 1 — `uv` (recommended)

```bash
brew install uv         # or follow https://docs.astral.sh/uv/getting-started/installation/
uv sync --extra dev     # creates .venv/ and resolves uv.lock
```

`make setup` runs this automatically when `uv` is on the `PATH`.

### Option 2 — plain `pip`

`pyproject.toml` is PEP 621–compliant, so it works directly with `pip`. A pinned `requirements.txt` (exported from `uv.lock`) is committed for fully reproducible installs:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # pinned versions matching uv.lock
pip install -e . --no-deps        # add the local packages
```

Or, for a non-pinned install straight from `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

`make setup` falls back to the pinned-`requirements.txt` path automatically when `uv` is not available.

Smoke test (either route):

```bash
python -c "from starixnet.models import STARIXNet"
```

To regenerate `requirements.txt` after editing `pyproject.toml`, run `make requirements` (requires `uv`).

## Commands

The Makefile auto-detects `uv`; when it's missing, it falls back to plain `pip` and assumes the venv is activated.

```bash
make setup          # uv sync --extra dev   (or pip install -r requirements.txt + -e .)
make setup.uv       # force the uv path
make setup.pip      # force the pip path
make requirements   # regenerate requirements.txt from uv.lock (uv only)

make test.unit      # run unit tests with coverage gate (>= 80%)
make lints          # ruff check + ruff format --check
make lints.fix      # ruff check --fix + ruff format
```

Invoking the tools directly works the same either way — prepend `uv run` if using uv, otherwise activate the venv first:

```bash
uv run pytest tests/test_starixnet.py -x        # uv route
# or, with the venv activated:
pytest tests/test_starixnet.py -x               # pip route
```

## Quick start

```python
import pandas as pd
from starixnet.models import STARIXNet

endog = pd.read_csv("data/endog.csv", index_col=[0]).interpolate()
exog = pd.read_csv("data/exog.csv", index_col=[0]).interpolate()

model = STARIXNet(
    n_temporal_embeddings=10080,
    enforced_periods=[10080, 1440],
    lagged_steps=30,
    forecast_steps=15,
    fit_multivariate=True,
)
model.fit(endog, exog=exog, loss_q=0.95, epochs=50, validation_split=0.2)

forecast = model.predict(endog.tail(30 + 15), exog=exog.tail(30 + 15), aggregation_q=0.5)
# forecast.shape == (samples, forecast_steps, endog_size)
```

The `applied_attributes` argument selects which decoder branches are built:

```python
STARIXNet(..., applied_attributes=["temporal", "autoregressive"])
```

## Architecture overview

| Decoder | Input | Mechanism |
| --- | --- | --- |
| Seasonal | scaled time index | `Σ amp·sin(freq·t + phase) + disp` with PSD-extracted frequencies |
| Temporal | scaled time index | embedding lookup over `t mod n_temporal_embeddings` + MLP |
| Autoregressive | differenced endog | per-target MLPs (univariate) or Conv1d + AdaptiveAvgPool (multivariate); integrated via `cumsum` |
| Exogenous | past exog values | Conv1d + AdaptiveMaxPool (multivariate) or Conv1d + MaxPool + per-target MLPs (univariate) |

Tensor convention throughout: `(samples, steps, features)`.

The benchmark models in `benchmark/models.py` follow the same `fit(endog, exog, …) / predict(endog, exog)` API and are implemented directly in PyTorch + Lightning (no `pytorch-forecasting` dependency).

## License

Apache License 2.0 — see [LICENSE](LICENSE).

### Third-party licenses

This project depends on several third-party packages. License information can be regenerated into `licenses/` with:

```bash
uv run pip-licenses --format=json > licenses/third-party-licenses.json
```

## Maintainer

Ahmed Abdulaal <ahmed.abdulaal@walmart.com>
