# PT-LiqAdj — Portfolio-aware Liquidity Adjustment (GNN + Transformer)

PT-LiqAdj is a small research/engineering project that explores **portfolio-aware** price/liquidity adjustments.  
The core idea: a bond’s realized delta depends not only on its own features, but also on the **composition of the traded portfolio** (issuer overlap, sector concentration, crowding). We capture this with a lightweight **GNN + Transformer** model and compare it to a plain **MLP baseline**.

---

## Key ideas (from the proposal)

- **Nodes are bonds.** Each bond has numeric features (size, coupon, days to maturity, …) and categorical attributes (sector, rating).
- **Local relational context.** We build simple relational groups (issuer, sector). A tiny message-passing block aggregates neighbor information to create **contextual node embeddings**.
- **Portfolio context.** Trades are portfolios (sequences/sets of bonds). A **cross-attention encoder** (Transformer) takes the target bond embedding as a query and the portfolio items as keys/values to produce a **portfolio-conditioned representation**.
- **Residual head.** A small MLP maps the fused representation to a predicted delta in bps. Optional temperature scaling can calibrate the mean/variance head if needed.
- **Baselines.** A simple MLP on per-trade, per-bond features ignores the portfolio composition to quantify the lift from portfolio awareness.

---

## Installation

```bash
# from project root
pip install -e .
# (optional) conda/venv should have PyTorch with CUDA if you want GPU training
````

---

## CLI overview

The project exposes a few small CLIs:

### 1) Simulate, validate, split, featurize, train, backtest (one-shot)

```
ptliq-run --config <yaml> [--workdir .]
```

* **Simulates** a small market, **validates** raw tables, **splits** time ranges, **featurizes**, **trains** the baseline MLP, then **backtests & reports**.
* Produces a report folder with figures (calibration, residual histogram) and a simple HTML report.

### 2) GNN vs Baseline experiment (portfolio-aware)

```
ptliq-exp --config configs/exp_gnn_vs_baseline.yaml --workdir .
```

* Simulates a larger dataset, builds graph+portfolio inputs, trains:

  * **Baseline** MLP (portfolio-agnostic),
  * **GNN+Transformer** portfolio-aware model,
* Evaluates both on the same test split and writes a JSON summary.

### 3) Individual tools (advanced / optional)

* `ptliq-sim` – generate synthetic `bonds.parquet` and `trades.parquet`.
* `ptliq-validate` – check schema/consistency of raw tables.
* `ptliq-split` – create train/val/test date ranges.
* `ptliq-feat` – build per-trade feature parquet files.
* `ptliq-train` – train baseline model.
* `ptliq-eval` – evaluate a trained baseline model on features.
* `ptliq-report` – render report graphics/HTML from backtest outputs.

(These are also orchestrated by `ptliq-run`.)

---

## Quickstart: end-to-end in one command

```bash
ptliq-run --config configs/exp_local.yaml --workdir .
```

Sample output (your numbers will vary but format is similar):

```
SIM → data/raw/sim
VAL passed → data/interim/validated/validation_report_YYYYMMDD-HHMMSS.json
SPLIT → data/interim/splits/<stamp>/ranges.json
FEAT → data/features/exp_local
TRAIN → models/exp_local (val_mae_bps≈0.20)
BACKTEST → reports/exp_local/backtest/<stamp>
REPORT → .../figures/calibration.png, residual_hist.png
```

---

## GPU usage

All training loops and the portfolio model accept `device` in config:

* `"cpu"`, `"cuda"`, or `"auto"` (use CUDA if available, otherwise CPU).
* Example (YAML):

```yaml
train:
  device: "auto"
```

---

## Example: GNN vs Baseline on simulated data

We include a ready-to-run experiment config:
`configs/exp_gnn_vs_baseline.yaml`

Run it:

```bash
ptliq-exp --config configs/exp_gnn_vs_baseline.yaml --workdir .
```

Sample results (from a recent run):

```
SIM → data/raw/sim
SPLIT → data/interim/splits/20251008-031225/ranges.json
BASELINE → best_epoch=2  val_mae=0.991  test_mae=6.160
GNN TRAIN → models/exp_gnn_vs_baseline_gnn (best_epoch=40)
GNN → best_epoch=40  test_mae=0.026

REPORT → reports/exp_gnn_vs_baseline/gnn_vs_baseline_20251008-031227.json
{
  "baseline": {
    "best_epoch": 2,
    "val_best_mae_bps": 0.9908409714698792,
    "test_mae_bps": 6.159795761108398
  },
  "gnn": {
    "best_epoch": 40,
    "test_mae_bps": 0.026281308382749557,
    "model_dir": "models/exp_gnn_vs_baseline_gnn"
  },
  "improvement_vs_baseline_pct": 99.57334123724225
}
```

**Interpretation.** The baseline MLP (no portfolio context) fails to explain the **portfolio-composition-dependent deltas** in our simulator, leading to a large test MAE. The portfolio-aware **GNN+Transformer** captures issuer/sector context and portfolio interactions, reducing MAE by ~**99.6%** in this synthetic setup.

---

## Configuration (minimal reference)

A minimal end-to-end config (used by `ptliq-run`):

```yaml
project:
  name: pt-liqadj
  seed: 42
  run_id: exp_local

paths:
  raw_dir: data/raw/sim
  interim_dir: data/interim
  features_dir: data/features
  models_dir: models
  reports_dir: reports

data:
  sim:
    n_bonds: 120
    n_days: 4
    providers: ["P1", "P2"]
    seed: 7

split:
  train_end: "2025-01-03"
  val_end: "2025-01-04"

train:
  device: "auto"          # "cpu" | "cuda" | "auto"
  max_epochs: 4
  batch_size: 512
  lr: 1e-3
  patience: 2
  hidden: [32]
  dropout: 0.0
  seed: 42
```

For the GNN experiment (`ptliq-exp`), the YAML also includes GNN hyper-parameters (depth, heads, dims). See `configs/exp_gnn_vs_baseline.yaml`.

---

## How the model is put together

* `ptliq/model/baseline.py` — MLP baseline (feature-only).
* `ptliq/model/backbone.py` — minimal message passing over issuer/sector groups for node embeddings.
* `ptliq/model/portfolio_encoder.py` — Transformer encoder with **cross-attention** from target bond to portfolio items.
* `ptliq/model/heads.py` — regression heads (mean, optional variance).
* `ptliq/model/model.py` — the **PortfolioResidualModel** that wires NodeEncoder → Backbone → PortfolioEncoder → Head.
* `ptliq/model/utils.py` — utilities and the `GraphInputs` container used by the GNN training loop.

Training:

* `ptliq/training/gnn_loop.py` — trains the GNN+Transformer model (early stopping on val MAE, saves best checkpoint).
* `ptliq/training/loop.py` — baseline MLP training/eval.

Pipelines/CLIs:

* `ptliq/cli/run.py` — end-to-end baseline pipeline (simulate→validate→split→features→train→backtest→report).
* `ptliq/cli/exp_gnn_vs_baseline.py` — **GNN vs baseline** experiment runner.

Visualization/Reporting:

* `ptliq/viz/report.py` — calibration, residual histograms, HTML.

---

## Testing

We ship a few sanity tests:

* `tests/test_model_components.py` — unit tests for the portfolio encoder & key components.
* `tests/test_gnn_xfmr_learns.py` — checks the GNN+Transformer can learn a synthetic **portfolio-dependent target** (GPU if available).
* Integration tests under `tests/test_cli_*` — basic end-to-end smoke tests for the baseline pipeline.

Run all:

```bash
python -m pytest -q
```

---

## Repro notes

* Seeds are set in configs and loops, but simulation randomness can still affect exact numbers.
* GPU vs CPU can change speed/ordering slightly; set `device: "cpu"` for bit-for-bit reproducibility on a single host.

---

## License

MIT (see `LICENSE`).

