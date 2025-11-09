# ptliq/cli/score.py  (PATCH: add DGTScorer path)
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ptliq.service.scoring import DGTScorer, MLPScorer, GRUScorer  # ensure DGTScorer is imported


def _read_rows(rows_json: str | None, rows_csv: str | None) -> List[Dict[str, Any]]:
    if rows_json:
        blob = json.loads(Path(rows_json).read_text())
        if isinstance(blob, dict):
            return [blob]
        return list(blob)
    if rows_csv:
        import pandas as pd
        df = pd.read_csv(rows_csv)
        return [r._asdict() if hasattr(r, "_asdict") else dict(r) for r in df.to_dict("records")]
    raise ValueError("Provide --rows-json or --rows-csv")

def _auto_model_type(model_dir: Path) -> str:
    # Heuristic: MV-DGT runs contain mvdgt_meta.json and view_masks.pt
    if (model_dir / "mvdgt_meta.json").exists() and (model_dir / "view_masks.pt").exists():
        return "dgt"
    # Fallbacks
    if (model_dir / "config.json").exists() and (model_dir / "ckpt.pt").exists():
        return "gru"
    return "mlp"

def main():
    ap = argparse.ArgumentParser(prog="ptliq-score", description="Generic scorer for packaged models")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--rows-json", default=None, help="Path to JSON array or single object")
    ap.add_argument("--rows-csv", default=None, help="Path to CSV with columns")
    ap.add_argument("--model-type", choices=["auto","dgt","gru","mlp"], default="auto")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    mtype = args.model_type if args.model_type != "auto" else _auto_model_type(model_dir)
    rows = _read_rows(args.rows_json, args.rows_csv)

    if mtype == "dgt":
        scorer = DGTScorer.from_dir(model_dir, device=args.device)
    elif mtype == "gru":
        scorer = GRUScorer.from_dir(model_dir, device=args.device)
    else:
        # MLP packaged zip/dir handled by MLPScorer; left as-is in your repo
        from ptliq.service.scoring import MLPScorer
        scorer = MLPScorer.from_dir(model_dir, device=args.device)

    y = scorer.score_many(rows)
    if args.out_csv:
        import pandas as pd
        out = pd.DataFrame(rows)
        out["pred_bps"] = np.asarray(y, dtype=np.float32)
        out.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")
    else:
        print(json.dumps(list(map(float, y))))

if __name__ == "__main__":
    main()
