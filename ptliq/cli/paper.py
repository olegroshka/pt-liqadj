# ptliq/cli/paper.py
from __future__ import annotations

import argparse
from pathlib import Path

from ptliq.paper.experiment import MakeDataConfig, make_data, score_scenarios


def _cmd_make_data(args: argparse.Namespace) -> None:
    cfg = MakeDataConfig(
        root=Path(args.root),
        seed=int(args.seed),
        n_nodes=args.n_nodes,
        n_days=args.n_days,
        out_model_dir=Path(args.model_dir) if args.model_dir else None,
        overwrite=not args.no_overwrite,
        simulate_args=tuple(args.simulate_args or ()),
        featurize_graph_args=tuple(args.feat_graph_args or ()),
        featurize_pyg_args=tuple(args.feat_pyg_args or ()),
        dgt_build_args=tuple(args.dgt_build_args or ()),
        dgt_train_args=tuple(args.dgt_train_args or ()),
    )
    meta = make_data(cfg)
    print("Wrote paper_meta.json:", meta)

def _cmd_score_scenarios(args: argparse.Namespace) -> None:
    out = score_scenarios(
        model_dir=Path(args.run_dir),
        out_dir=Path(args.out),
        n_warm_groups=int(args.n_warm_groups),
        reps_per_group=int(args.reps_per_group),
        seed=int(args.seed),
    )
    for k, v in out.items():
        print(f"{k}: {v}")

def main() -> None:
    p = argparse.ArgumentParser(prog="ptliq-paper", description="Paper helpers: make data & score scenarios")
    sub = p.add_subparsers(dest="cmd", required=True)

    # make-data
    p1 = sub.add_parser("make-data", help="Use existing CLIs to simulate→featurize→build→train")
    p1.add_argument("--root", required=True, help="Root folder to write the run (e.g., paper_runs/exp001)")
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--n-nodes", dest="n_nodes", type=int, default=None)
    p1.add_argument("--n-days", dest="n_days", type=int, default=None)
    p1.add_argument("--model-dir", default=None, help="Where to put the trained model; default <root>/models/dgt")
    p1.add_argument("--no-overwrite", action="store_true", help="Do not clean subfolders under root")
    # Pass-throughs to your CLIs (remain optional)
    p1.add_argument("--simulate-args", nargs="*", default=None)
    p1.add_argument("--feat-graph-args", nargs="*", default=None)
    p1.add_argument("--feat-pyg-args", nargs="*", default=None)
    p1.add_argument("--dgt-build-args", nargs="*", default=None)
    p1.add_argument("--dgt-train-args", nargs="*", default=None)
    p1.set_defaults(func=_cmd_make_data)

    # score-scenarios
    p2 = sub.add_parser("score-scenarios", help="Replay scenarios and write CSVs for figures")
    p2.add_argument("--run-dir", required=True, help="Trained MV-DGT run dir (contains ckpt.pt)")
    p2.add_argument("--out", required=True, help="Folder to receive CSVs")
    p2.add_argument("--n-warm-groups", type=int, default=3)
    p2.add_argument("--reps-per-group", type=int, default=2)
    p2.add_argument("--seed", type=int, default=100)
    p2.set_defaults(func=_cmd_score_scenarios)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
