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
* `ptliq-train-baseline` – train baseline model.
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


---

## Developer setup and installation

- Python 3.10+ recommended.
- Install in editable mode:

```
pip install -e .[dev]
```

Notes:
- For GPU training, install PyTorch with CUDA first (see https://pytorch.org/get-started/locally/), then run the command above.
- Torch Geometric wheels are pulled automatically via `pip` for many CUDA/torch combos; if you hit platform issues, consult https://pytorch-geometric.readthedocs.io/ for the exact install command for your environment.

### Useful Makefile targets (optional)
If you prefer `make` helpers (when available in your environment):

```
make lint   # ruff, mypy
make test   # pytest -q
```

---

## CLI commands (from pyproject.toml)
The project exposes several CLI entry points after installation. Highlights:

- Data lifecycle:
  - `ptliq-simulate` — simulate raw data (`data/raw/sim`).
  - `ptliq-validate` — validate schema/consistency of raw tables.
  - `ptliq-split` — create chronological train/val/test ranges.
  - `ptliq-featurize` — feature pipelines:
    - `ptliq-featurize graph` — build graph artifacts (nodes/edges, portfolio weights, market features).
    - `ptliq-featurize pyg` — convert to PyG tensors and feature meta.
  - `ptliq-explore` — quick statistics/plots for parquet files.
  - `ptliq-pyg-explore` — inspect PyG features run directory.

- Training:
  - `ptliq-gat-train` — train the GATv2-based portfolio model on PyG features.
  - `ptliq-dgt-build` — prepare MV-DGT samples/masks from trades + graph + PyG.
  - `ptliq-dgt-train` — train MV-DGT from the prepared workdir.

- Orchestration / misc:
  - `ptliq-run`, `ptliq-exp` — end-to-end or experiment runners.
  - `ptliq-start-tensorboard`, `ptliq-stop-tensorbord` — helper commands to manage TensorBoard.

Full list lives in `pyproject.toml` under `[project.scripts]`.

---

## Project goal (brief)
Portfolio-aware liquidity adjustment: predict per-bond price impact/residual not only from the bond’s own features but also from portfolio composition and market context. We compare portfolio-agnostic baselines to portfolio-aware GNN/attention models and provide reproducible pipelines end-to-end.

---

## Practical pipelines with example commands
Below are reproducible, copy-pastable snippets to get you from raw data to trained models. Paths assume running from the project root and write artifacts under `data/` and `models/`.

### 1) Simulate and validate raw data

```
# Generate a small synthetic dataset
ptliq-simulate --outdir data/raw/sim

# Validate schema/consistency
ptliq-validate --rawdir data/raw/sim
```

Optional: explore the generated tables.

```
ptliq-explore data/raw/sim/bonds.parquet --correlations --plots --pdf
ptliq-explore data/raw/sim/trades.parquet --correlations --plots --pdf
```

### 2) Build graph + PyG features

```
# Graph construction (relations + portfolio weights + market features)
ptliq-featurize graph \
  --bonds data/raw/sim/bonds.parquet \
  --trades data/raw/sim/trades.parquet \
  --outdir data/graph \
  --cotrade-q 0.85 \
  --cotrade-topk 20

# Convert to PyG tensors
ptliq-featurize pyg \
  --graph-dir data/graph \
  --outdir data/pyg
```

You can inspect PyG features with:

```
ptliq-pyg-explore --features-run-dir data/pyg --pdf
```

### 3) Train the GAT model (GNN + attention)

Minimal run on simulated data (CUDA if available):

```
ptliq-gat-train \
  --features-run-dir data/features/sim1000 \
  --trades data/raw/sim/trades.parquet \
  --graph-dir data/graph \
  --outdir models/liquidity/exp_sim1001_gatv2 \
  --config configs/gat.default.yaml \
  --seed 7 \
  --tb \
  --tb-log-dir models/exp_sim1001_gatv2/tb \
  --device cuda
```

Notes:
- `--features-run-dir` should point to a PyG features run (e.g., `data/pyg`). If you use a different folder structure (e.g., `data/features/some_run`), point there accordingly.
- Override hyper-parameters via CLI or `configs/gat.default.yaml`.

### 4) Build and train MV-DGT

Prepare the MV-DGT working directory (samples + masks):

```
ptliq-dgt-build \
  --trades-path data/raw/sim/trades.parquet \
  --graph-dir data/graph \
  --pyg-dir data/pyg \
  --outdir data/mvdgt/exp001
```

Train MV-DGT:

```
ptliq-dgt-train \
  --workdir data/mvdgt/exp001 \
  --pyg-dir data/pyg \
  --epochs 20 \
  --lr 1e-3 \
  --batch-size 1024 \
  --outdir models/dgt_8
```

---

## Paper/report workflow: generate data, tables, and figures
This repository includes a small helper CLI to reproduce the data and figures used in the project report/paper. It stitches together existing commands into three simple steps.

Prerequisites:
- Install the project (pip install -e .). GPU is optional; training uses CUDA if available.
- Run all commands from the project root.

### Step 1 — Create a paper run (simulate → featurize → build → train)
This prepares synthetic data, graph + PyG features, builds MV-DGT samples/masks, and trains the MV-DGT model.

```
ptliq-paper make-data --root paper_runs/exp001
```

What it does under the hood (roughly):
- ptliq-simulate → <root>/data/raw/sim
- ptliq-featurize graph → <root>/data/graph
- ptliq-featurize pyg → <root>/data/pyg
- ptliq-dgt-build → <root>/data/mvdgt/exp001
- ptliq-dgt-train → <root>/models/dgt

It also writes a convenience manifest:
- <root>/paper_meta.json with paths like raw_dir, graph_dir, pyg_dir, work_dir, model_dir.

Useful options:
- --seed 42 — simulation and training seed (default 42)
- --n-nodes, --n-days — override simulator size/horizon
- --model-dir <path> — custom output location for trained model
- --no-overwrite — do not clean subfolders under <root> before running
- Pass-throughs to low-level CLIs (optional):
  - --simulate-args ...
  - --feat-graph-args ...
  - --feat-pyg-args ...
  - --dgt-build-args ...
  - --dgt-train-args ... (e.g., --epochs 30 --lr 5e-3 --batch-size 512 --seed 42 --device auto)

Example (matches our internal repro):
```
ptliq-paper make-data --root paper_runs/exp001
```

### Step 2 — Score paper scenarios and write CSV tables
Given the trained run directory, export CSVs used by the figures.

```
ptliq-paper score-scenarios \
  --run-dir paper_runs/exp001/models/dgt \
  --out paper/tables
```

This will produce (paths under --out):
- warm_scenarios.csv
- cold_scenarios.csv
- portfolio_drift.csv
- ablation.csv
- negative_drag.csv
- parity.csv

### Step 3 — Render figures (PNG/PDF) from tables
```
ptliq-paper make-figures \
  --tables-dir paper/tables \
  --out paper/figs
```

Outputs include (both .png and .pdf by default):
- fig_warm_size_elasticity, fig_warm_side_flip, fig_warm_time_roll
- fig_cold_size_elasticity, fig_cold_side_flip
- fig_portfolio_drift_hist
- fig_ablation
- fig_negative_drag

Notes:
- You can choose formats with --formats, e.g.:
  - ptliq-paper make-figures --tables-dir paper/tables --out paper/figs --formats png pdf svg
- Training uses device="auto"; to force CPU or CUDA, pass via --dgt-train-args (e.g., --device cpu or --device cuda).

---

## TensorBoard: start/stop and where to look

Most training commands can log to TensorBoard. Typical locations are under each model’s `tb/` subdirectory (e.g., `models/exp_sim1001_gatv2/tb`, `models/mvdgt/tb`).

Start TensorBoard with our helper:

```
ptliq-start-tensorboard --logdir models --port 6006
```

Stop it later (helper name spelling as in `pyproject.toml`):

```
ptliq-stop-tensorbord --port 6006
```

Or use the native command directly:

```
tensorboard --logdir models --port 6006
```

Open http://localhost:6006 in your browser.

---

## Serving the API and the demo website

The project ships with two CLIs that let you serve the scoring API and a tiny demo website.
Defaults are chosen so you can run both with zero arguments.

### Start the FastAPI scoring service

```
ptliq-serve
```

Defaults:
- package: `serving/tmp_model` (a tiny local model for smoke testing)
- host: `127.0.0.1`
- port: `8011`
- device: `cpu`

Override as needed, e.g. to point at a packaged model zip:

```
ptliq-serve --package serving/packages/my_run.zip --host 0.0.0.0 --port 8011
```

Stop the server (reads the pidfile and terminates the process):

```
ptliq-serve stop
# or explicitly (useful if you changed the port):
ptliq-serve stop --port 8011
```

Tips:
- If you get a message about an existing pidfile, either stop the old process or start with `--force`:
  - `ptliq-serve --force`

### Start the demo website (Gradio UI)

The website accepts a JSON payload like:

```
{"rows":[{"isin":"US1","f_a":1.2,"f_b":-0.7},{"isin":"US2","f_a":0.0,"f_b":3.3}]}
```

and displays a grid with columns `Portfolio Id | Isin | Portfolio Liquidity Impact (bps)`, including multi-select filters. Positive values indicate extra execution drag (worse), negative values indicate relief (better); this is interpreted by side (buy: +bps = higher paid price; sell: +bps = lower received price).

Start the site (defaults to the local API server):

```
ptliq-web
```

Defaults:
- api-url: `http://127.0.0.1:8011`
- host: `127.0.0.1`
- port: `7861`
- open browser: yes (disable with `--no-open-browser`)

Examples:

```
# Disable auto-opening a browser
ptliq-web --no-open-browser

# Start on a different port and point to a remote API
ptliq-web --api-url http://my-api-host:8011 --port 9000
```

Stop the website:

```
ptliq-web stop
# or explicitly by port
ptliq-web stop --port 7861
```

If you ever encounter a page stuck on "Loading…":
- Stop the web process (`ptliq-web stop`) and start again.
- Open the site in an Incognito/Private window to avoid stale service-worker cache.
- Ensure the API (`ptliq-serve`) is running and healthy at the configured `--api-url` (`/health` returns JSON).

---

## Troubleshooting tips
- If `torch-geometric` complains about incompatible wheels, reinstall it matching your Torch/CUDA versions (see official docs), then reinstall this project with `pip install -e .`.
- On CPU-only hosts, pass `--device cpu` to training CLIs or set `device: "cpu"` in configs.
- Use `ptliq-explore` and `ptliq-pyg-explore` to sanity-check inputs before training.


---

## Using real TRACE data with MV‑DGT

If you plan to train/serve MV‑DGT on real markets (TRACE Enhanced + vendor reference/evaluated prices), see the step‑by‑step adapter guide:

- docs/TRACE_to_MV_DGT_ADAPTER.md — maps TRACE/security‑master fields to the minimal trades.parquet and bonds.parquet schemas, includes a reference pandas adapter snippet, and shows how to run featurization/build steps.
