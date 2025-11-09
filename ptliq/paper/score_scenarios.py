# paper/score_scenarios.py
from __future__ import annotations
import argparse
from pathlib import Path
from ptliq.paper.experiment import score_scenarios

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-warm-groups", type=int, default=3)
    ap.add_argument("--reps-per-group", type=int, default=2)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    out = score_scenarios(
        model_dir=Path(args.run_dir),
        out_dir=Path(args.out),
        n_warm_groups=args.n_warm_groups,
        reps_per_group=args.reps_per_group,
        seed=args.seed,
    )
    print(out)

if __name__ == "__main__":
    main()
