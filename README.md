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

---

## MV‑DGT architecture — detailed, presentation‑level overview

This section explains the Multi‑View Differential Graph Transformer (MV‑DGT) used in this repo for portfolio‑aware signals. It is written for a first‑time reader and ties the concepts to the exact code paths in `ptliq/model/mv_dgt.py` and dataset tooling under `ptliq/features` and `ptliq/training`.

Relevant files:
- Model: `ptliq/model/mv_dgt.py` (class `MultiViewDGT`)
- Dataset builder (views/masks): `ptliq/features/build_mvdgt_dataset.py`
- Training loop: `ptliq/training/mvdgt_loop.py`
- Example config: `configs/mvdgt.default.yaml`
- Diagrams: `models/dgt_demo/mv_dgt_concept_*.{png,jpg}`

### What MV‑DGT solves (intuitive)

MV‑DGT learns a per‑trade signal for a focal asset (the “anchor”) while being explicitly aware of:
- Multi‑view relations in a market graph (issuer/sector structure, portfolio co‑membership, and correlation‑based neighbors).
- Portfolio context: a strict leave‑one‑out (LOO) prototype around the anchor to subtract common portfolio drift and keep stock/bond‑specific residuals.
- Optional sample‑level features (market/trade) and optional within‑portfolio self‑attention capturing interactions among items in the same batch portfolio group.

The model is compact and each part is gated, making ablations and diagnostics straightforward.

### Inputs and core shapes

Notation below follows the implementation in `mv_dgt.py`:
- `x: [N, x_dim]` — per‑node features for the full universe (e.g., static or day‑specific graph node features).
- `edge_index: [2, E]`, `edge_weight: [E, 1]` — global graph connectivity and initial weights (registered as buffers in the model).
- Per‑view boolean masks (buffers): `mask_struct`, `mask_port`, `mask_corr_global`, `mask_corr_local` — same length `E` as `edge_index` columns.
- `anchor_idx: [B]` — indices of focal nodes (the assets we predict on in this batch).
- `pf_gid: [B]` — portfolio group id per sample; `-1` means “no portfolio context for this sample”.
- `port_ctx: dict` — flattened portfolio context, see below.
  - `port_nodes_flat: [L]` — node ids for all portfolio line items.
  - `port_w_signed_flat: [L]` — signed weights (e.g., side‑signed size) per line.
  - `port_w_abs_flat: [L]` (optional) — absolute weights; if missing, `abs(signed)` is used.
  - `port_len: [G]` — item counts per portfolio group id.
- Optional per‑sample features:
  - `market_feat: [B, mkt_dim]` — market context features aligned by `date_idx`.
  - `trade_feat: [B, trade_dim]` — side/size/urgency or other trade‑level signals.

All shapes and semantics are exercised in unit tests under `tests/test_mvdgt_e2e.py` and by the training loop.

### How the multi‑view relations are built

The graph edges are prepared offline by your featurizer and stored as a PyG artifact (`pyg_graph.pt`) along with an `edge_type` vector that labels each edge with a relation id. The dataset builder groups those relations into 4 views and saves boolean masks `view_masks.pt` aligned to `edge_index` length (`E`). See `ptliq/features/build_mvdgt_dataset.py`:

- Structural view (`struct`): relation names in `REL_STRUCT = {"ISSUER_SIBLING","SECTOR","RATING_NEAR","CURVE_BUCKET","CURRENCY"}`.
  - SECTOR relations connect nodes sharing the same sector code.
  - ISSUER_SIBLING connects bonds from the same issuer.
  - RATING_NEAR can connect neighbors within a narrow rating band.
  - CURVE_BUCKET and CURRENCY are other structural affinities.
- Portfolio co‑membership (`port`): `REL_PORT = {"COTRADE_CO","COTRADE_X"}` — co‑occurred in the same portfolio/day. The exact meaning of “CO” vs “X” depends on featurizer choices (same account or cross‑account, etc.).
- Correlation — global (`corr_global`): `REL_CG = {"PCC_GLOBAL","MI_GLOBAL"}`.
- Correlation — local (`corr_local`): `REL_CL = {"PCC_LOCAL","MI_LOCAL"}`.

The helper `_make_view_masks(edge_type, id_sets)` maps relation ids to these masks; the training loop validates that mask lengths match `E`.

#### How sector relations are encoded (single or multiple taxonomies)

- In the simplest case, each node has a single sector code (`node_to_sector`) and the featurizer emits edges labeled `SECTOR` between all pairs that share the sector. Those edges land in the structural view via the `SECTOR` relation id.
- If you have multiple sector taxonomies (e.g., GICS level 1/2/3, or issuer industry vs sector), emit separate relation names like `SECTOR_L1`, `SECTOR_L2`, `INDUSTRY` and list them inside `REL_STRUCT`. The dataset builder will include those relation ids in the `struct` mask. Nodes with multiple sector memberships naturally produce multiple edges — one per taxonomy match — which the model can learn to weigh via per‑view attention and gates.
- Edge weights: whatever the featurizer sets (e.g., 1.0 for unweighted structural links or a frequency/strength score). MV‑DGT standardizes edge weights per view on the fly to remove scale mismatch before message passing.

#### Where masks and edges come from in practice

- `pyg_graph.pt` contains `edge_index` and `edge_weight` (directed edges) and an `edge_type` vector.
- `view_masks.pt` is a dict `{view_name: BoolTensor[E]}` built with `_build_view_ids` + `_make_view_masks` in `build_mvdgt_dataset.py`.
- The training loop `_load_pyg_and_view_masks` (in `ptliq/training/mvdgt_loop.py`) loads these and puts them on the chosen device. It also writes a copy under the run directory for reproducibility.

### Component breakdown and responsibilities

The code maps 1‑to‑1 to the blocks below.

1) Shared node encoder (per node)
- `enc: Linear(x_dim→hidden) → ReLU → Dropout → Linear(hidden→hidden)`; then `LayerNorm` (`norm0`).
- Lifts raw `x` into the shared embedding space `H: [N, hidden]`.

2) Multi‑view graph encoder (two graph layers; per‑view TransformerConv + differential fusion)
- For each view `v ∈ {struct, port, corr_global, corr_local}` we have a `TransformerConv(hidden→hidden)` in each layer: `conv1[v]` and `conv2[v]`.
- Each layer has learnable per‑view scalar gates: `g1_logit[v]` and `g2_logit[v]` → `sigmoid` to get `g[v] ∈ (0,1)`.
- Correlation edges are softly down‑weighted by a learnable `corr_gate` (initialized to −1, so `sigmoid(−1)≈0.27`).
- Before each message pass for non‑struct views, edge weights are standardized within that view: `ew = (ew - mean) / (std + 1e-6)`.
- Differential fusion (per layer) treats `struct` as baseline and learns deviations for other views:

```
h = x + g[struct]*h_s
      + g[port]*(h_port - h_s)
      + g[corr_global]*(h_cg - h_s)
      + g[corr_local]*(h_cl - h_s)
```

This lets the model say “use structural messages unless another view brings demonstrably different, gated information.”

3) Anchor gather (per sample)
- The “anchor” is simply the focal asset for which we produce a prediction in this batch entry.
- Implementation: `z_anchor_pre = H[anchor_idx]` via `_gather_anchor`.

4) Portfolio LOO prototype and residual fusion (strict LOO; signless in forward)
- We compute per‑sample strict LOO vectors in the same `H` space using only co‑portfolio neighbors (never the anchor itself), with efficient groupwise accumulation over `port_ctx`.
- In `compute_samplewise_portfolio_vectors_loo(H, anchor_idx, pf_gid, port_ctx)` we return two `B×hidden` tensors:
  - `V_abs` — absolute prototype (weighted average of others’ `H` vectors using absolute weights).
  - `V_sgn` — signed prototype, factorized as `(sum_signed_others / sum_abs_others) * V_abs` so its direction follows `V_abs` but magnitude carries signed mass.
- Both are L2‑normalized by default (with care not to erase signed‑mass semantics of `V_sgn` when it’s zero).
- In the forward path we intentionally enforce signless behavior for residual subtraction to avoid signed leakage: feed `[V_abs, 0]` to a projection and subtract a gated residual from the anchor:

```
pf_feat = concat(V_abs, zeros_like(V_abs))
z_anchor = z_anchor_pre - sigmoid(pf_gate) * pf_proj(pf_feat)
```

This centers the anchor on a portfolio‑conditioned residual while remaining strictly LOO.

5) Optional sample‑level context encoders
- Market MLP (`mkt_enc`) and Trade MLP (`trade_enc`): `Linear→ReLU→(Dropout)→Linear` → `hidden`.
- If enabled and provided, these yield `z_mkt` and `z_trade` appended to the head input (or used in portfolio attention).

6) Optional within‑portfolio self‑attention (off by default)
- Build a token per sample by concatenating enabled parts among `{z_anchor, z_trade?, z_mkt?}` → project to `Hb`.
- Pack tokens by `pf_gid` into padded groups (optionally capped by `max_portfolio_len`) and run a `TransformerEncoder`.
- Fuse back as either:
  - Residual mode: `z_anchor ← z_anchor + sigmoid(portfolio_gate) * fuse(ctx)`; or
  - Concat mode: append `sigmoid(portfolio_gate) * fuse(ctx)` to the head input.

7) Optional portfolio head (signless absolute prototype)
- A small MLP on `V_abs` adds a gated, clean portfolio‑prototype branch to the head input, if enabled.

8) Regression head and deterministic negative drag
- The head concatenates enabled branches `[z_anchor,(z_mkt?),(z_trade?),(pf_head?),(attn_ctx?)] → MLP → ŷ`.
- Optional “negative drag” biases outputs away from the co‑portfolio direction (signless) using the learned H‑space:

```
Vh_abs = compute_samplewise_portfolio_vectors_loo(H, anchor_idx, pf_gid, port_ctx)
cos_h = |cos( normalize(z_anchor_pre), Vh_abs )|
yhat = yhat - pf_drag_coef * cos_h
```

This enforces conservative, portfolio‑aware signals even if the head tries to re‑align with the portfolio axis.

9) Attention diagnostics (optional)
- If enabled, per‑view attention head mean/std are captured from `TransformerConv` layers per view and per layer, for introspection.

### How inputs are blended to extract portfolio‑conditioned signals

- Multi‑view fusion learns “what counts as a neighbor” via per‑view attention and scalar gates; `struct` is the anchor baseline and other views inject controlled deviations.
- The anchor embedding is explicitly de‑biased by subtracting a gated projection of the strict‑LOO absolute prototype (signless), preventing leakage of co‑portfolio drift while keeping informative structure.
- Market/trade branches provide exogenous context per example; optional within‑portfolio attention captures interactions among items that share a `pf_gid` in the current batch.
- The head operates on this residual‑refined representation; optional negative‑drag further discourages co‑portfolio alignment in the output.

### Exact anchors, portfolios, and port_ctx representation

- Anchor (“what we predict for”) is the sample’s `node_id` at `anchor_idx[i]`.
- `pf_gid[i]` ties samples that belong to the same portfolio group (e.g., same portfolio/day). `-1` means “no portfolio” for that sample.
- `port_ctx` is global to the batch step and contains flattened per‑group lines:
  - `port_nodes_flat: [L]` — node ids of all lines across groups;
  - `port_w_signed_flat: [L]`, `port_w_abs_flat: [L]` — signed and absolute weights per line;
  - `port_len: [G]` — counts per group; used to segment the flat arrays.
- The LOO logic carefully removes all contributions from the anchor node in its own group (even if it appears multiple times) before forming prototypes and normalizing.

### Configuration knobs you will likely touch

In YAML (`configs/mvdgt.default.yaml`) and in `MVDGTModelConfig`:
- `hidden`, `heads`, `dropout` — width and attention multiplicity of graph/MLPs.
- `use_portfolio`, `use_market`, `trade_dim` — toggle portfolio residual, market/trade branches.
- `use_portfolio_attn`, `portfolio_attn_layers/heads/hidden/mode/gate`, `max_portfolio_len` — within‑portfolio interaction encoder.
- `use_pf_head`, `pf_head_hidden` — auxiliary portfolio‑prototype branch.
- `use_negative_drag` — deterministic portfolio aversion term at the output.

### How to build and train MV‑DGT

1) Build dataset artifacts (graph views + per‑trade samples):

```
ptliq-mvdgt-build \
  --trades-path data/mvdgt/trades.parquet \
  --graph-dir  data/graph \
  --pyg-dir    data/pyg \
  --outdir     data/mvdgt/exp001
```

This creates: `outdir/samples.parquet`, `outdir/view_masks.pt`, and `outdir/mvdgt_meta.json`, and expects `pyg_dir/pyg_graph.pt` with `edge_index`, `edge_weight`, and `edge_type`.

2) Train the model:

```
ptliq-mvdgt-train --workdir data/mvdgt/exp001 --pyg-dir data/pyg --epochs 20 --device auto
```

You can provide a YAML to override defaults: `--config configs/mvdgt.default.yaml`.

3) (Optional) Export ONNX:

```
ptliq-mvdgt-export-onnx --workdir data/mvdgt/exp001 --outdir models/mvdgt_onnx
```

### Diagrams

We include several visual summaries under `models/dgt_demo/`:
- `mv_dgt_concept_v1.png` — early conceptual diagram.
- `mv_dgt_concept_ppt4x3.png` and `mv_dgt_concept_square.png` — slide/thumbnail layouts.

These are presentation‑only visuals; the definitive behavior is the code and equations above.
