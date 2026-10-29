# STARIXNet

Encoder–decoder neural network for multivariate, multi-attribute time-series forecasting. The model decomposes a forecast into independent decoder branches — **S**easonal, **T**emporal, **A**uto**R**egressive **I**ntegrated, and e**X**ogenous — then aggregates them via a quantile reduction at prediction time. Six benchmark challengers (SARIMA, DeepAR, TFT, CrossLinear, TimeXer, iTransformer) are included; see [Benchmarks](#benchmarks).

> **Citing STARIXNet:** if you use this work, please cite the [accompanying paper](#how-to-cite) (arXiv:2606.07565).

## Repository layout

```
starixnet/      # the proposed STARIXNet model (models.py), training loop (utilities.py), and quantile loss (losses.py)
benchmark/      # SARIMA, DeepAR, TFT, CrossLinear, TimeXer, iTransformer (+ LSTM, SeasonalNaive) in PyTorch + Lightning
notebooks/      # tutorial notebook and the script that generates it
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

The autoregressive decoder's backbone is selectable with `ar_encoder="conv"` (default, Conv1d + AdaptivePool) or `ar_encoder="transformer"` (self-attention over the lagged window).

## Tutorial notebook

[`notebooks/starixnet_tutorial.ipynb`](notebooks/starixnet_tutorial.ipynb) walks through the full workflow on a synthetic service-monitoring dataset (CPU / TPS / memory targets with traffic, deployment, error-rate, network, and uptime exogenous signals). It covers data generation, training, per-decoder forecast inspection, the quantile-aggregation sweep, and save/load.

```bash
# install Jupyter + matplotlib (kept out of the core dependencies)
uv sync --extra notebook                # or: pip install -e ".[notebook]"
uv run jupyter lab notebooks/starixnet_tutorial.ipynb
```

The notebook is regenerated from a source script (`notebooks/_build_tutorial.py`) — keeps the diff reviewable as Python rather than raw JSON.

## Architecture overview

| Decoder | Input | Mechanism |
| --- | --- | --- |
| Seasonal | scaled time index | `Σ amp·sin(freq·t + phase) + disp` with PSD-extracted frequencies |
| Temporal | scaled time index | embedding lookup over `t mod n_temporal_embeddings` + MLP |
| Autoregressive | differenced endog | per-target MLPs (univariate) or Conv1d + AdaptiveAvgPool (multivariate), or a transformer encoder with `ar_encoder="transformer"`; integrated via `cumsum` |
| Exogenous | past exog values | Conv1d + AdaptiveMaxPool (multivariate) or Conv1d + MaxPool + per-target MLPs (univariate) |

Tensor convention throughout: `(samples, steps, features)`.

**Drift.** The seasonal and temporal branches are stationary by construction. For targets with monotonic drift (memory growth, capacity creep), pass a monotonic covariate such as `uptime_minutes` through `exog` and use a higher `aggregation_q` (0.7–0.9) so the trend-aware branches aren't diluted by the median.

## Benchmarks

STARIXNet is compared against **6 challengers**, chosen to be mutually dissimilar, to cover the current state-of-the-art families, and (wherever the family allows it) to consume the same past-only exogenous signals as STARIXNet's exogenous decoder. SARIMA and TFT are fixed members of the set; the other four are revisited as the literature moves. Models considered and excluded include Informer, Autoformer, FEDformer, TimesNet, TimeMixer, DLinear and PatchTST: each is either superseded by or architecturally overlapping with a selected model, or cannot consume exogenous signals.

| Model | Family | Exogenous signals | Why it is in the set |
| --- | --- | --- | --- |
| **SARIMA** | Classical statistical | accepted, silently dropped | Non-deep anchor; answers "is deep learning needed at all?"; ARIMA is the customary baseline in cloud-workload forecasting |
| **DeepAR** | Recurrent, probabilistic | encoder only; endog-only autoregressive rollout | Principled recurrent baseline with probabilistic output; subsumes a plain LSTM |
| **TFT** | Recurrent + attention, covariate-aware | encoder only | The established covariate-aware deep model, the most direct comparator for STARIXNet's exogenous decoder |
| **CrossLinear** | Linear / patch-MLP, exog-aware | cross-correlation Conv1d over `[endog \| exog]` | Linear-family "is the complexity justified?" baseline built for exogenous variables; replaces DLinear (channel-independent, cannot use exog) |
| **TimeXer** | Patch transformer + exog cross-attention | exog variate tokens via cross-attention | Strongest time-axis transformer with exogenous support; replaces PatchTST (channel-independent, cannot use exog) |
| **iTransformer** | Variate attention | exog as extra variate tokens | Attends across variates rather than time; most relevant to the multivariate framing |

Also in the library but **not** part of the paper's set: `LSTM` (kept for inference-service compatibility; DeepAR subsumes it) and `SeasonalNaive` (training-free "repeat last season" floor; season = `max(enforced_periods)` or the PSD-detected period).

### Shared contract

All benchmarks share STARIXNet's `fit(endog, exog, …)` / `predict(endog, exog, …)` API and are implemented directly in PyTorch + Lightning (no `pytorch-forecasting` dependency). For the deep benchmarks:

- **Exogenous signals are past-only**, like the targets at encoding time. They are consumed only when fitting with `exog_as_unknown_reals=True`; the default `False` ignores them (pre-refactor behaviour). No model ever treats exog as known future input.
- **`predict` returns one forecast per full `lagged_steps` window**: an input of `n` rows yields `n - lagged_steps + 1` forecasts of shape `(forecast_steps, endog_size)`, and the last forecast is anchored on the newest observation. With `skip_current_step=True` the horizon starts one step after the step following the window.

```python
from benchmark.models import TimeXer

model = TimeXer(lagged_steps=30, forecast_steps=15, skip_current_step=True)
model.fit(endog, exog=[exog_1, exog_2], exog_as_unknown_reals=True, epochs=50, validation_split=0.2)
forecast = model.predict(endog.tail(30), exog=[e.tail(30) for e in (exog_1, exog_2)])
# forecast.shape == (1, 15, endog_size)
```

### References

<details>
<summary>Original papers and recent cloud-workload uses of each benchmark (click to expand)</summary>

Original papers, followed by recent uses of the model in cloud-resource / workload forecasting where we could find them. For CrossLinear and TimeXer we found no cloud-workload application yet, so recent general benchmark usage is listed instead.

- **SARIMA**
  - Box, G. E. P., Jenkins, G. M., Reinsel, G. C., Ljung, G. M. *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley, 2015.
  - Calheiros, R. N., Masoumi, E., Ranjan, R., Buyya, R. Workload Prediction Using ARIMA Model and Its Impact on Cloud Applications' QoS. *IEEE Transactions on Cloud Computing*, 3(4), 2015. [link](https://www.researchgate.net/publication/270735403_Workload_Prediction_Using_ARIMA_Model_and_Its_Impact_on_Cloud_Applications'_QoS)
  - Using LSTM and SARIMA Models to Forecast Cluster CPU Usage. arXiv:2007.08092, 2020. [link](https://arxiv.org/abs/2007.08092)
  - Forecasting Workload in Cloud Computing: Towards Uncertainty-Aware Predictions and Transfer Learning. arXiv:2303.13525, 2023 (ARIMA as the standard evaluation baseline). [link](https://arxiv.org/abs/2303.13525)
- **DeepAR**
  - Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks. *International Journal of Forecasting*, 36(3), 2020. [link](https://arxiv.org/abs/1704.04110)
  - Wang, S., Sun, Y., Shi, X., et al. Full Scaling Automation for Sustainable Development of Green Data Centers. *IJCAI 2023* (DeepAR baseline for microservice workload / CPU-utilization forecasting). [link](https://arxiv.org/abs/2305.00706)
- **TFT**
  - Lim, B., Arık, S. Ö., Loeff, N., Pfister, T. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting*, 37(4), 2021. [link](https://arxiv.org/abs/1912.09363)
  - Ba, A., Harsha, P., Subramanian, C. Leveraging Interpretability in the Transformer to Automate the Proactive Scaling of Cloud Resources. arXiv:2409.03103, 2024 (TFT for microservice latency forecasting driving resource scaling). [link](https://arxiv.org/abs/2409.03103)
  - CITRAS: Covariate-Informed Transformer for Time Series Forecasting. arXiv:2503.24007, 2025 (TFT as a covariate-aware baseline). [link](https://arxiv.org/abs/2503.24007)
- **CrossLinear**
  - Zhou, P., Liu, Y., Liang, J., Song, Q., Li, X. CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables. *KDD 2025*. [link](https://arxiv.org/abs/2505.23116) · [code](https://github.com/mumiao2000/CrossLinear)
- **TimeXer**
  - Wang, Y., Wu, H., Dong, J., et al. TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables. *NeurIPS 2024*. [link](https://arxiv.org/abs/2402.19072)
  - Used as the exogenous-variable baseline in CrossLinear (*KDD 2025*) and Timer-XL (*ICLR 2025*). [link](https://arxiv.org/abs/2410.04803)
- **iTransformer**
  - Liu, Y., Hu, T., Zhang, H., et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *ICLR 2024*. [link](https://arxiv.org/abs/2310.06625)
  - Chen, J., Ye, H., Jiang, F., et al. Fremer: Lightweight and Effective Frequency Transformer for Workload Forecasting in Cloud Services. arXiv:2507.12908, 2025 (iTransformer baseline on ByteDance cloud-service workloads). [link](https://arxiv.org/abs/2507.12908)
  - Cloud Resource Workload Forecasting Method Based on the MST-iTransformer Model. *Future Internet*, 18(9), 2026 (iTransformer variant on the Alibaba cluster trace). [link](https://www.mdpi.com/1999-5903/18/9/448)

</details>

## How to cite

If you use this work in any way, please cite the accompanying paper:

### STARIXNet (arXiv preprint):

    Abdulaal, A., Aytekin, M., Srinivasan, T. K., Lancewicki, T. (2026).
    STARIXNet: Multivariate and Multi-attribute Deep Learning Approach to Real-Time Resource Allocation in Cloud Platforms.
    arXiv preprint arXiv:2606.07565.
click [here](https://arxiv.org/abs/2606.07565) to access the above publication.

BibTeX:

```bibtex
@article{abdulaal2026starixnet,
  title={STARIXNet: Multivariate and Multi-attribute Deep Learning Approach to Real-Time Resource Allocation in Cloud Platforms},
  author={Abdulaal, Ahmed and Aytekin, Maruf and Srinivasan, Thilaga Kumaran and Lancewicki, Tomer},
  journal={arXiv preprint arXiv:2606.07565},
  year={2026}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

### Third-party licenses

This project depends on several third-party packages. License information can be regenerated into `licenses/` with:

```bash
uv run pip-licenses --format=json > licenses/third-party-licenses.json
```

## Maintainer

Ahmed Abdulaal <AhmedR.Abdulaal@gmail.com>
