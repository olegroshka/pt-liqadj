# paper/make_data.py
from __future__ import annotations
import argparse
from pathlib import Path
from ptliq.paper.experiment import MakeDataConfig, make_data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-nodes", type=int, default=None)
    ap.add_argument("--n-days", type=int, default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--no-overwrite", action="store_true")
    args = ap.parse_args()
    cfg = MakeDataConfig(
        root=Path(args.root),
        seed=args.seed,
        n_nodes=args.n_nodes,
        n_days=args.n_days,
        out_model_dir=Path(args.model_dir) if args.model_dir else None,
        overwrite=not args.no_overwrite,
    )
    meta = make_data(cfg)
    print(meta)

if __name__ == "__main__":
    main()
