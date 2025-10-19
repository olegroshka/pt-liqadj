from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import torch

# Matplotlib optional
try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAVE_MPL = True
except Exception:
    plt = None  # type: ignore
    _HAVE_MPL = False


def validate_pyg(data_path: Path, outdir: Path) -> Dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Explicitly set weights_only=False for compatibility with PyTorch >=2.6 safe loading defaults
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    problems = []
    if int(data.edge_index.min()) < 0 or int(data.edge_index.max()) >= int(data.num_nodes):
        problems.append("edge_index has out-of-range node ids")
    try:
        finite_x = torch.isfinite(data.x).all().item()
    except Exception:
        finite_x = True
    if not finite_x:
        problems.append("x has NaN/Inf")

    deg = (
        torch.bincount(data.edge_index[0], minlength=data.num_nodes)
        + torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    )
    isolates = int((deg == 0).sum())
    rel_counts: Dict[int, int] = {}
    if hasattr(data, "edge_type"):
        et = data.edge_type.detach().cpu().numpy().tolist()
        uniq, cnt = np.unique(et, return_counts=True)
        for k, v in zip(uniq, cnt):
            rel_counts[int(k)] = int(v)

    report = {
        "num_nodes": int(data.num_nodes),
        "num_edges_directed": int(data.edge_index.size(1)),
        "x_dim": int(data.x.size(1)) if hasattr(data, "x") else 0,
        "isolates": isolates,
        "degree": {
            "min": int(deg.min()),
            "median": float(deg.float().median()),
            "mean": float(deg.float().mean()),
            "max": int(deg.max()),
        },
        "relation_id_counts": rel_counts,
        "problems": problems,
    }
    (outdir / "dataset_report.json").write_text(json.dumps(report, indent=2))

    if _HAVE_MPL:
        plt.figure()
        plt.hist(deg.detach().cpu().numpy(), bins=20)
        plt.title("Directed degree histogram")
        plt.xlabel("degree")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "degree_hist.png", dpi=160)
        plt.close()

        # Feature inspection: first two numeric features if present
        if hasattr(data, "x"):
            x = data.x.detach().cpu().numpy()
            if x.shape[1] >= 1:
                plt.figure()
                plt.hist(x[:, 0], bins=20)
                plt.title("Feature[0]")
                plt.tight_layout()
                plt.savefig(outdir / "feature0_hist.png", dpi=160)
                plt.close()
            if x.shape[1] >= 2:
                plt.figure()
                plt.hist(x[:, 1], bins=20)
                plt.title("Feature[1]")
                plt.tight_layout()
                plt.savefig(outdir / "feature1_hist.png", dpi=160)
                plt.close()

    return report
