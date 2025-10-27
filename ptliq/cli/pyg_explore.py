# ptliq/cli/pyg_explore.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import os
import json

import numpy as np
import torch
import typer
from rich import print
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)


def _suppress_mpl_max_open_warning() -> None:
    """Suppress matplotlib warning about many open figures.

    We intentionally generate many pages/figures for the PDF report, but we do
    close figures after saving. To avoid noisy RuntimeWarning emitted during
    figure creation, lower/disable the threshold via rcParams.
    """
    try:
        import matplotlib as mpl  # type: ignore
        # 0 disables the warning entirely
        mpl.rcParams["figure.max_open_warning"] = 0
    except Exception:
        pass


def _make_table_pages(
    headers: List[str],
    rows: List[List[str]],
    title: str,
    rows_per_page: int = 28,
    col_widths: Optional[List[float]] = None,
):
    import matplotlib.pyplot as plt  # type: ignore
    figs = []
    n = len(rows)
    pages = max(1, (n + rows_per_page - 1) // rows_per_page)
    for p in range(pages):
        start = p * rows_per_page
        end = min(n, (p + 1) * rows_per_page)
        chunk = rows[start:end]
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.set_title(f"{title} (page {p+1}/{pages})" if pages > 1 else title)
        table = ax.table(cellText=chunk, colLabels=headers, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        # Try to fit table nicely
        table.scale(1.0, 1.2)
        # Column widths if provided
        if col_widths is not None:
            for i, w in enumerate(col_widths):
                table.auto_set_column_width(col=i)
        fig.tight_layout()
        figs.append(fig)
    return figs


def _load_pyg(path: Path):
    try:
        from torch_geometric.data import Data, HeteroData  # noqa: F401
    except Exception as e:
        raise typer.BadParameter("torch-geometric is required. Install with: pip install torch-geometric") from e
    data = torch.load(path, weights_only=False)
    return data


def _safe_numpy(x: torch.Tensor) -> np.ndarray:
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return np.asarray(x)


def _degree_undirected(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    deg = np.zeros(num_nodes, dtype=np.int64)
    # edge_index shape (2, E)
    src = edge_index[0]
    dst = edge_index[1]
    # Ignore self loops for degree (optional): here we count them
    np.add.at(deg, src, 1)
    return deg


def _unique_undirected_edges(edge_index: np.ndarray) -> np.ndarray:
    # Return mask over columns picking one per undirected pair (i<j)
    s = edge_index[0]
    d = edge_index[1]
    lo = np.minimum(s, d)
    hi = np.maximum(s, d)
    undirected_pairs = np.stack([lo, hi], axis=1)
    # Remove self pairs separately (lo==hi) will remain but unique will keep one
    # Use structured array trick for uniqueness
    view = undirected_pairs.view([('lo', undirected_pairs.dtype), ('hi', undirected_pairs.dtype)])
    _, idx = np.unique(view, return_index=True)
    mask = np.zeros(edge_index.shape[1], dtype=bool)
    mask[idx] = True
    return mask


def _build_nx_graph(edge_index: np.ndarray, edge_weight: Optional[np.ndarray] = None, edge_type: Optional[np.ndarray] = None):
    try:
        import networkx as nx  # type: ignore
    except Exception as e:
        raise typer.BadParameter("networkx is required for subgraph visualization. pip install networkx") from e
    G = nx.Graph()
    E = edge_index.shape[1]
    for e in range(E):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])
        if u == v:
            continue
        w = float(edge_weight[e]) if edge_weight is not None and e < len(edge_weight) else 1.0
        rel = int(edge_type[e]) if edge_type is not None and e < len(edge_type) else -1
        # for undirected unique edges add once with max weight among duplicates; keep the relation of the strongest edge
        if G.has_edge(u, v):
            if w > G[u][v].get('weight', 0.0):
                G[u][v]['weight'] = w
                G[u][v]['rel'] = rel
        else:
            G.add_edge(u, v, weight=w, rel=rel)
    return G


def _plot_degree_hist(deg: np.ndarray, bins: int = 50):
    import matplotlib.pyplot as plt  # type: ignore
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(deg, bins=bins, color="#4C72B0", edgecolor="white")
    ax.set_title("Node degree distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("#Nodes")
    fig.tight_layout()
    return fig


def _plot_edge_weight_hist(w: np.ndarray, bins: int = 50):
    import matplotlib.pyplot as plt  # type: ignore
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(w, bins=bins, color="#55A868", edgecolor="white")
    ax.set_title("Edge weight distribution")
    ax.set_xlabel("Weight")
    ax.set_ylabel("#Edges")
    fig.tight_layout()
    return fig


def _plot_relation_bar(rel_ids: np.ndarray, rel_names: Optional[List[str]]):
    import matplotlib.pyplot as plt  # type: ignore
    vals, counts = np.unique(rel_ids, return_counts=True)
    labels = [str(int(v)) for v in vals]
    if rel_names is not None:
        # best effort mapping (assume id order)
        labels = [rel_names[int(v)] if int(v) < len(rel_names) else str(int(v)) for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(vals)), counts, color="#C44E52")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title("Edges per relation type")
    ax.set_ylabel("#Edges (directed)")
    fig.tight_layout()
    return fig


def _plot_adjacency_sparsity(edge_index: np.ndarray, num_nodes: int, sample_nodes: int = 500):
    import matplotlib.pyplot as plt  # type: ignore
    rng = np.random.default_rng(17)
    if num_nodes > sample_nodes:
        nodes = np.sort(rng.choice(num_nodes, size=sample_nodes, replace=False))
        mask = np.isin(edge_index[0], nodes) & np.isin(edge_index[1], nodes)
        ei = edge_index[:, mask]
        # remap nodes to 0..k-1
        mapping = {n: i for i, n in enumerate(nodes.tolist())}
        src = np.vectorize(mapping.get)(ei[0])
        dst = np.vectorize(mapping.get)(ei[1])
        ei = np.vstack([src, dst])
        n = sample_nodes
    else:
        ei = edge_index
        n = num_nodes
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ei[0], ei[1], s=1, c="#222222")
    ax.set_xlim(-1, n)
    ax.set_ylim(-1, n)
    ax.set_title("Adjacency pattern (sampled)")
    ax.set_xlabel("src")
    ax.set_ylabel("dst")
    fig.tight_layout()
    return fig


def _plot_subgraph(
    G,
    nodes: List[int],
    title: str,
    relation_names: Optional[List[str]] = None,
    max_rel_legend: int = 8,
    node_types: Optional[Dict[int, int]] = None,
):
    import matplotlib.pyplot as plt  # type: ignore
    try:
        import networkx as nx  # type: ignore
    except Exception as e:
        raise typer.BadParameter("networkx is required for subgraph visualization. pip install networkx") from e

    H = G.subgraph(nodes)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    pos = nx.spring_layout(H, seed=17)

    # --- Edge styling by relation type ---
    # collect relation ids present
    rel_ids = []
    for u, v, d in H.edges(data=True):
        rel_ids.append(int(d.get("rel", -1)))
    rel_ids = np.array(rel_ids, dtype=int) if len(rel_ids) else np.array([], dtype=int)

    # Count relations and keep top-k for legend
    rel_order = []
    if len(rel_ids):
        uniq, cnt = np.unique(rel_ids, return_counts=True)
        order = uniq[np.argsort(-cnt)]
        rel_order = order[:max_rel_legend].tolist()

    # color palette
    import matplotlib as mpl  # type: ignore
    cmap = mpl.cm.get_cmap("tab20", max(10, len(rel_order) + 1))
    rel_to_color: Dict[int, any] = {}
    for i, rid in enumerate(rel_order):
        rel_to_color[int(rid)] = cmap.colors[i] if hasattr(cmap, "colors") else cmap(i)
    other_color = (0.6, 0.6, 0.6, 0.7)

    # Build edge lists per relation bucket
    edges_by_rel: Dict[str, list] = {"other": []}
    for u, v, d in H.edges(data=True):
        rid = int(d.get("rel", -1))
        key = rid if rid in rel_to_color else "other"
        edges_by_rel.setdefault(key, []).append((u, v, d))

    # Optional edge width scaled by weight (normalize within subgraph)
    weights = [float(d.get("weight", 1.0)) for _, _, d in H.edges(data=True)]
    if len(weights) > 0:
        w_arr = np.array(weights, dtype=float)
        w_min, w_max = float(np.nanmin(w_arr)), float(np.nanmax(w_arr))
        def norm_w(x: float) -> float:
            if not np.isfinite(x):
                return 0.8
            if w_max <= w_min + 1e-9:
                return 1.0
            val = 0.5 + 2.5 * (x - w_min) / (w_max - w_min)
            return float(np.clip(val, 0.5, 3.0))
    else:
        def norm_w(x: float) -> float:
            return 1.0

    # Draw edges per group for proper legend
    for key, edgelist in edges_by_rel.items():
        if key == "other" and len(rel_to_color) == 0 and len(edgelist) == 0:
            continue
        color = other_color if key == "other" else rel_to_color[int(key)]
        el = [(u, v) for (u, v, _) in edgelist]
        widths = [norm_w(float(d.get("weight", 1.0))) for (_, _, d) in edgelist]
        if el:
            nx.draw_networkx_edges(H, pos=pos, edgelist=el, edge_color=[color]*len(el), width=widths, alpha=0.9, ax=ax)

    # --- Node styling (ensure distinct from edges) ---
    node_border_color = "white"
    if node_types is not None and len(node_types) > 0:
        # map node ids to types for nodes present
        types_present = sorted({int(node_types.get(n, -1)) for n in H.nodes()})
        # pastel-like palette for nodes to differ from edge tab20
        node_colors = ["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]
        t_to_color = {t: node_colors[i % len(node_colors)] for i, t in enumerate(types_present)}
        for t in types_present:
            nn = [n for n in H.nodes() if int(node_types.get(n, -1)) == t]
            if nn:
                nx.draw_networkx_nodes(
                    H, pos=pos, nodelist=nn, node_size=70,
                    node_color=[t_to_color[t]]*len(nn), linewidths=0.6, edgecolors=node_border_color, ax=ax
                )
        node_leg_items = [(f"node type {t}", t_to_color[t]) for t in types_present]
    else:
        # homogeneous nodes: neutral gray distinct from colored edges
        nx.draw_networkx_nodes(
            H, pos=pos, nodelist=list(H.nodes()), node_size=70,
            node_color="#6e6e6e", linewidths=0.6, edgecolors=node_border_color, ax=ax
        )
        node_leg_items = [("nodes", "#6e6e6e")]

    # Build legend handles
    handles = []
    from matplotlib.patches import Patch  # type: ignore
    for rid, color in rel_to_color.items():
        label = f"{relation_names[rid]}" if (relation_names is not None and 0 <= rid < len(relation_names)) else f"rel {rid}"
        handles.append(Patch(color=color, label=label))
    if len(edges_by_rel.get("other", [])) > 0:
        handles.append(Patch(color=other_color, label="other relations"))
    # node legend (single entry if homogeneous)
    for lbl, col in node_leg_items[:5]:
        handles.append(Patch(color=col, label=lbl))

    ax.legend(handles=handles, loc="best", fontsize=7, frameon=True)
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    return fig


def _write_pdf(
    outdir: Path,
    stem: str,
    overview_lines: Optional[List[str]],
    figures: List[Tuple[str, object]],
    metrics_lines: Optional[List[str]] = None,
    hints_lines: Optional[List[str]] = None,
    schema_lines: Optional[List[str]] = None,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
    except Exception:
        print("[yellow]matplotlib not installed; cannot write PDF. pip install matplotlib seaborn[/yellow]")
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{stem}__pyg_report.pdf"

    def add_text_page(title: str, lines: List[str]):
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.02, 0.98, "\n".join([title, "", *lines]), va="top", family="monospace", fontsize=9)
        pdf_out.savefig(fig)
        plt.close(fig)

    with PdfPages(pdf_path) as pdf_out:
        if overview_lines:
            add_text_page("PyG dataset overview", overview_lines)
        if schema_lines:
            add_text_page("Feature schema", schema_lines)
        if metrics_lines:
            add_text_page("Graph metrics", metrics_lines)
        if hints_lines:
            add_text_page("How to read these visuals", hints_lines)
        for title, fig in figures:
            # Avoid adding a suptitle to prevent overlap with plot titles
            try:
                fig.tight_layout()
            except Exception:
                pass
            pdf_out.savefig(fig)
            try:
                import matplotlib.pyplot as plt  # type: ignore
                plt.close(fig)
            except Exception:
                pass
    return pdf_path


@app.command("pyg-explore")
def app_main(
    features_run_dir: Optional[Path] = typer.Option(None, help="Folder with pyg_graph.pt and feature_meta.json"),
    pt_path: Optional[Path] = typer.Option(None, help="Direct path to pyg_graph.pt (overrides features_run_dir)"),
    outdir: Optional[Path] = typer.Option(None, help="Output dir for report/plots (default: <features_run_dir>/reports or reports/pyg)"),
    degree_bins: int = typer.Option(50, help="Bins for degree histogram"),
    topk_high: int = typer.Option(30, help="Top-K nodes by degree for subgraph"),
    topk_low: int = typer.Option(30, help="Bottom-K nodes by degree (incl. isolated) for subgraph"),
    pdf: bool = typer.Option(True, help="Write multi-page PDF report"),
):
    console = Console()

    # Suppress matplotlib max-open warning for many report pages
    _suppress_mpl_max_open_warning()

    # Resolve input path
    if pt_path is None:
        if features_run_dir is None:
            features_run_dir = Path(os.getenv("PTLIQ_DEFAULT_PYG_DIR", "data/pyg"))
        pt_path = features_run_dir / "pyg_graph.pt"
        if outdir is None:
            # prefer colocated reports dir
            outdir = (features_run_dir / "reports")
    else:
        if outdir is None:
            outdir = Path("reports/pyg")

    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise typer.BadParameter(f"pyg_graph.pt not found at: {pt_path}")

    meta: Dict | None = None
    meta_path = (pt_path.parent / "feature_meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = None

    data = _load_pyg(pt_path)

    # Currently we expect torch_geometric.data.Data with fields: x, edge_index, edge_type, edge_weight, num_nodes
    num_nodes = int(getattr(data, 'num_nodes', 0) or (data.x.size(0) if hasattr(data, 'x') else 0))
    edge_index = _safe_numpy(data.edge_index).astype(np.int64)
    edge_type = _safe_numpy(data.edge_type) if hasattr(data, 'edge_type') else None
    edge_weight = _safe_numpy(data.edge_weight) if hasattr(data, 'edge_weight') else None

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise typer.BadParameter("edge_index must have shape (2, E)")

    E_dir = edge_index.shape[1]
    deg = _degree_undirected(edge_index, num_nodes)

    # undirected unique edges assuming symmetric storage
    undirected_mask = _unique_undirected_edges(edge_index)
    E_undir = int(undirected_mask.sum())
    # self-loops and isolated nodes
    self_loops = int(np.sum(edge_index[0] == edge_index[1]))
    isolated = int(np.sum(deg == 0))

    possible_undir = num_nodes * (num_nodes - 1) // 2
    density = (E_undir / possible_undir) if possible_undir > 0 else float('nan')
    sparsity = 1.0 - density if not np.isnan(density) else float('nan')

    # Console overview
    table = Table(title="PyG dataset overview")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Nodes", f"{num_nodes:,}")
    table.add_row("Edges (directed)", f"{E_dir:,}")
    table.add_row("Edges (undirected unique)", f"{E_undir:,}")
    table.add_row("Self-loops", f"{self_loops:,}")
    table.add_row("Isolated nodes", f"{isolated:,}")
    table.add_row("Density (undirected)", f"{density:.6f}")
    table.add_row("Sparsity (1-density)", f"{sparsity:.6f}")
    console.print(table)

    # Feature info
    x_dim = int(data.x.size(1)) if hasattr(data, 'x') and data.x is not None else 0
    issuer_index = getattr(data, 'issuer_index', None)
    corr_mask = getattr(data, 'corr_edge_mask', None)

    # Build figures
    figures: List[Tuple[str, object]] = []
    try:
        fig = _plot_degree_hist(deg, bins=degree_bins)
        figures.append(("Degree histogram", fig))
    except Exception as e:
        print(f"[yellow]Failed to create degree histogram: {e}[/yellow]")

    if edge_weight is not None:
        try:
            fig = _plot_edge_weight_hist(edge_weight, bins=50)
            figures.append(("Edge weight histogram", fig))
        except Exception as e:
            print(f"[yellow]Failed to create edge weight histogram: {e}[/yellow]")

    rel_names = None
    if meta and isinstance(meta.get("relations"), list):
        rel_names = meta["relations"]

    if edge_type is not None:
        try:
            fig = _plot_relation_bar(edge_type.astype(int), rel_names)
            figures.append(("Edges per relation", fig))
        except Exception as e:
            print(f"[yellow]Failed relation bar: {e}[/yellow]")

    try:
        fig = _plot_adjacency_sparsity(edge_index, num_nodes, sample_nodes=500)
        figures.append(("Adjacency sparsity (sample)", fig))
    except Exception as e:
        print(f"[yellow]Failed adjacency plot: {e}[/yellow]")

    # Small subgraphs
    try:
        et_undir = edge_type[undirected_mask] if edge_type is not None else None
        ew_undir = edge_weight[undirected_mask] if edge_weight is not None else None
        ei_undir = edge_index[:, undirected_mask]
        G = _build_nx_graph(ei_undir, ew_undir, et_undir)
        # try optional node types (if the Data carries it)
        node_types = None
        if hasattr(data, 'node_type') and getattr(data, 'node_type') is not None:
            try:
                node_types_np = _safe_numpy(getattr(data, 'node_type')).astype(int).tolist()
                node_types = {i: int(t) for i, t in enumerate(node_types_np)}
            except Exception:
                node_types = None
        # top-k high degree (overall)
        deg_sorted = np.argsort(-deg)
        top_nodes = deg_sorted[: min(topk_high, len(deg_sorted))].tolist()
        if len(top_nodes) > 0:
            fig = _plot_subgraph(
                G,
                top_nodes,
                f"Top-{len(top_nodes)} degree subgraph (colored by relation; width∝weight)",
                relation_names=rel_names,
                node_types=node_types,
            )
            figures.append(("Top-K subgraph (overall)", fig))
        # bottom-k (including isolated)
        low_sorted = np.argsort(deg)
        low_nodes = low_sorted[: min(topk_low, len(low_sorted))].tolist()
        if len(low_nodes) > 0:
            fig = _plot_subgraph(
                G,
                low_nodes,
                f"Bottom-{len(low_nodes)} degree subgraph (colored by relation; width∝weight)",
                relation_names=rel_names,
                node_types=node_types,
            )
            figures.append(("Bottom-K subgraph (overall)", fig))

        # --- New: per-relation subgraphs (top relations by frequency) ---
        if et_undir is not None and len(et_undir) == ei_undir.shape[1]:
            rel_top_n = 6
            vals, counts = np.unique(et_undir.astype(int), return_counts=True)
            order = vals[np.argsort(-counts)]
            # helper to label relations by name when available
            def rel_label(rid_int: int) -> str:
                if rel_names is not None and 0 <= int(rid_int) < len(rel_names):
                    return str(rel_names[int(rid_int)])
                return f"{rid_int}"

            for rid in order[:rel_top_n]:
                mask_r = (et_undir.astype(int) == int(rid))
                ei_r = ei_undir[:, mask_r]
                if ei_r.shape[1] == 0:
                    continue
                ew_r = ew_undir[mask_r] if ew_undir is not None else None
                et_r = et_undir[mask_r] if et_undir is not None else None
                G_r = _build_nx_graph(ei_r, ew_r, et_r)
                # degrees within this relation
                deg_r = np.zeros(num_nodes, dtype=int)
                np.add.at(deg_r, ei_r[0].astype(int), 1)
                np.add.at(deg_r, ei_r[1].astype(int), 1)
                # top-k for this relation
                deg_sorted_r = np.argsort(-deg_r)
                top_nodes_r = [int(n) for n in deg_sorted_r[: min(topk_high, len(deg_sorted_r))] if deg_r[n] > 0]
                if top_nodes_r:
                    label = rel_label(int(rid))
                    fig = _plot_subgraph(
                        G_r,
                        top_nodes_r,
                        f"Top-{len(top_nodes_r)} subgraph — relation {label}",
                        relation_names=rel_names,
                        node_types=node_types,
                    )
                    title = f"Top-K subgraph (relation={label})"
                    figures.append((title, fig))
                # bottom-k for this relation: nodes with the smallest positive degree in this relation
                pos_nodes = np.where(deg_r > 0)[0]
                if len(pos_nodes) > 0:
                    order_low = pos_nodes[np.argsort(deg_r[pos_nodes])]
                    low_nodes_r = [int(n) for n in order_low[: min(topk_low, len(order_low))]]
                    if low_nodes_r:
                        label = rel_label(int(rid))
                        fig = _plot_subgraph(
                            G_r,
                            low_nodes_r,
                            f"Bottom-{len(low_nodes_r)} subgraph — relation {label}",
                            relation_names=rel_names,
                            node_types=node_types,
                        )
                        title = f"Bottom-K subgraph (relation={label})"
                        figures.append((title, fig))

            # --- Ensure CO_TRADE relations are always plotted (by name) ---
            try:
                if rel_names is not None and len(rel_names) > 0:
                    cotrade_ids = [i for i, nm in enumerate(rel_names) if isinstance(nm, str) and ("COTRADE" in nm.upper() or "CO_TRADE" in nm.upper())]
                else:
                    cotrade_ids = []
            except Exception:
                cotrade_ids = []

            for rid in cotrade_ids:
                # skip if already included above
                if rid in order[:rel_top_n]:
                    continue
                mask_r = (et_undir.astype(int) == int(rid))
                ei_r = ei_undir[:, mask_r]
                if ei_r.shape[1] == 0:
                    continue
                ew_r = ew_undir[mask_r] if ew_undir is not None else None
                et_r = et_undir[mask_r] if et_undir is not None else None
                G_r = _build_nx_graph(ei_r, ew_r, et_r)
                # degrees within this relation
                deg_r = np.zeros(num_nodes, dtype=int)
                np.add.at(deg_r, ei_r[0].astype(int), 1)
                np.add.at(deg_r, ei_r[1].astype(int), 1)
                # top-k for this relation
                deg_sorted_r = np.argsort(-deg_r)
                top_nodes_r = [int(n) for n in deg_sorted_r[: min(topk_high, len(deg_sorted_r))] if deg_r[n] > 0]
                label = rel_label(int(rid))
                if top_nodes_r:
                    fig = _plot_subgraph(
                        G_r,
                        top_nodes_r,
                        f"Top-{len(top_nodes_r)} subgraph — relation {label}",
                        relation_names=rel_names,
                        node_types=node_types,
                    )
                    title = f"Top-K subgraph (relation={label})"
                    figures.append((title, fig))
                # bottom-k for this relation
                pos_nodes = np.where(deg_r > 0)[0]
                if len(pos_nodes) > 0:
                    order_low = pos_nodes[np.argsort(deg_r[pos_nodes])]
                    low_nodes_r = [int(n) for n in order_low[: min(topk_low, len(order_low))]]
                    if low_nodes_r:
                        fig = _plot_subgraph(
                            G_r,
                            low_nodes_r,
                            f"Bottom-{len(low_nodes_r)} subgraph — relation {label}",
                            relation_names=rel_names,
                            node_types=node_types,
                        )
                        title = f"Bottom-K subgraph (relation={label})"
                        figures.append((title, fig))
    except Exception as e:
        print(f"[yellow]Failed to produce subgraphs: {e}[/yellow]")

    # Optional: feature correlations when manageable
    if hasattr(data, 'x') and data.x is not None and x_dim > 1 and num_nodes > 1 and x_dim <= 200:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            x_np = _safe_numpy(data.x).astype(float)
            # Subsample rows for speed when huge
            if num_nodes > 5000:
                idx = np.random.default_rng(17).choice(num_nodes, size=5000, replace=False)
                x_np = x_np[idx]
            corr = np.corrcoef(x_np, rowvar=False)

            # Feature names from meta (fallback to x0..x{C-1})
            feat_names: List[str] = []
            try:
                if meta and isinstance(meta.get("node_feature_schema"), dict):
                    cols = meta["node_feature_schema"].get("columns")
                    if isinstance(cols, list):
                        feat_names = [str(c) for c in cols]
            except Exception:
                feat_names = []
            C = corr.shape[0]
            if not feat_names or len(feat_names) != C:
                feat_names = [f"x{i}" for i in range(C)]

            # Build heatmap figure with tick labels using feature names (thinned if many)
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
            fig.colorbar(im, ax=ax)
            max_labels = 60
            step = max(1, int(np.ceil(C / max_labels)))
            ticks = list(range(0, C, step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([feat_names[i] for i in ticks], rotation=90, fontsize=6)
            ax.set_yticks(ticks)
            ax.set_yticklabels([feat_names[i] for i in ticks], fontsize=6)
            title = "Node feature correlation heatmap"
            if step > 1:
                title += f" (labels every {step})"
            ax.set_title(title)
            ax.set_xlabel("Feature")
            ax.set_ylabel("Feature")
            fig.tight_layout()
            figures.append(("Feature correlations", fig))

            # Also include a small table figure with top correlation pairs using feature names
            try:
                iu = np.triu_indices_from(corr, k=1)
                pairs = []
                for i, j in zip(iu[0], iu[1]):
                    pairs.append((feat_names[i], feat_names[j], float(corr[i, j])))
                pairs.sort(key=lambda t: abs(t[2]), reverse=True)
                top_k = pairs[:12]
                # Build a compact table page
                import matplotlib.pyplot as plt  # type: ignore
                fig_pairs, ax_pairs = plt.subplots(figsize=(7.5, 3 + 0.25*len(top_k)))
                ax_pairs.axis('off')
                headers = ["Feature A", "Feature B", "corr"]
                rows = [[a, b, f"{r:+.3f}"] for a, b, r in top_k]
                table = ax_pairs.table(cellText=rows, colLabels=headers, loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.0, 1.2)
                ax_pairs.set_title("Top correlation pairs (features)")
                fig_pairs.tight_layout()
                figures.append(("Top correlation pairs (features)", fig_pairs))
            except Exception:
                pass
        except Exception as e:
            print(f"[yellow]Failed feature correlation heatmap: {e}[/yellow]")

    overview = [
        f"Path: {pt_path}",
        f"Nodes: {num_nodes:,}",
        f"Edges directed: {E_dir:,} | undirected unique: {E_undir:,}",
        f"Self-loops: {self_loops:,} | Isolated nodes: {isolated:,}",
        f"Density (undir) = {density:.8f}  |  Sparsity = {sparsity:.8f}",
        f"x_dim: {x_dim}",
        f"issuer_index: {'yes' if issuer_index is not None else 'no'}  |  corr_edge_mask: {'yes' if corr_mask is not None else 'no'}",
    ]

    metrics = [
        f"deg min/mean/p50/max: {int(deg.min()) if len(deg)>0 else 0} / {deg.mean():.3f} / {np.median(deg):.3f} / {int(deg.max()) if len(deg)>0 else 0}",
    ]

    if edge_weight is not None and len(edge_weight) > 0:
        ew = edge_weight.astype(float)
        metrics.append(f"weight min/p50/mean/max: {np.nanmin(ew):.4g} / {np.nanmedian(ew):.4g} / {np.nanmean(ew):.4g} / {np.nanmax(ew):.4g}")

    # Print brief schema-like to console and build schema page lines for PDF
    schema = Table(title="Feature schema (x)")
    schema.add_column("Field")
    schema.add_column("Value")
    schema.add_row("x_dim", str(x_dim))

    schema_lines: List[str] = []
    schema_lines.append("This page summarizes node feature schema and related metadata.")
    schema_lines.append("")
    schema_lines.append(f"x_dim: {x_dim}")

    if meta and isinstance(meta.get("node_feature_schema"), dict):
        nfs = meta["node_feature_schema"]
        cols = nfs.get("columns")
        if isinstance(cols, list):
            shown = 80
            cols_str = ", ".join(map(str, cols[:shown]))
            if len(cols) > shown:
                cols_str += " ..."
            schema.add_row("x columns", ", ".join(map(str, cols[:40])) + (" ..." if len(cols) > 40 else ""))
            schema_lines.append(f"x columns ({len(cols)}): {cols_str}")
        iss = nfs.get("issuer_index")
        if iss is not None:
            schema.add_row("issuer_index", str(iss))
            schema_lines.append(f"issuer_index field present: {bool(iss)}")
    else:
        # Fallback: infer issuer_index presence from Data attribute
        schema_lines.append(f"issuer_index tensor attached: {'yes' if issuer_index is not None else 'no'}")

    # Relations
    if rel_names:
        schema.add_row("relations", ", ".join(rel_names))
        rel_str = ", ".join(rel_names[:40]) + (" ..." if len(rel_names) > 40 else "")
        schema_lines.append(f"relations ({len(rel_names)}): {rel_str}")

    # Additional meta sections if present
    if meta and isinstance(meta, dict):
        if isinstance(meta.get("feature_dims"), dict):
            fd = meta["feature_dims"]
            schema_lines.append("")
            schema_lines.append("feature_dims:")
            for k in ["num_nodes","num_edges_directed","x_dim","num_relations"]:
                if k in fd:
                    schema_lines.append(f"  - {k}: {fd[k]}")
        if isinstance(meta.get("market_context"), dict):
            mc = meta["market_context"]
            schema_lines.append("")
            schema_lines.append("market_context:")
            for k in ["num_days","num_features","index_file"]:
                if k in mc:
                    schema_lines.append(f"  - {k}: {mc[k]}")
        if isinstance(meta.get("portfolio_context"), dict):
            pc = meta["portfolio_context"]
            schema_lines.append("")
            schema_lines.append("portfolio_context:")
            for k in ["num_groups","total_lines","index_file"]:
                if k in pc:
                    schema_lines.append(f"  - {k}: {pc[k]}")

    console.print(schema)

    # Guidance/hints on interpretation
    hints = [
        "Degree histogram: Heavy right tail or hubs may cause over-smoothing or training collapse in GNNs.",
        "  - Watch for: many nodes with degree=0 (isolates), extremely high-degree hubs.",
        "Edge weight histogram: Skewed or multi-modal weights may destabilize message passing.",
        "  - Watch for: extreme outliers; consider clipping or normalizing.",
        "Edges per relation: Severe imbalance can bias relational attention.",
        "  - Watch for: one relation dominating; consider re-weighting or sampling.",
        "Adjacency pattern (sample): Blocky structures suggest communities; diagonal bands suggest ordering artifacts.",
        "  - Watch for: unexpected dense blocks (leakage?) or almost empty patterns (over-sparsity).",
        "Top-K subgraph: Reveals hubs/bridges.",
        "  - Watch for: star-shaped hubs; consider capping neighbors or using dropout.",
        "Bottom-K subgraph: Reveals isolates/fragmentation.",
        "  - Watch for: many isolates; consider adding features/relations to connect components.",
        "Feature correlation heatmap: Near ±1 correlations indicate redundancy/leakage.",
        "  - Watch for: features perfectly tracking targets or each other; consider PCA/feature drop.",
    ]

    # Prepend well-formatted tables for overview and schema to the figures sequence
    try:
        overview_headers = ["Metric", "Value"]
        overview_rows = [
            ["Path", str(pt_path)],
            ["Nodes", f"{num_nodes:,}"],
            ["Edges (directed)", f"{E_dir:,}"],
            ["Edges (undirected unique)", f"{E_undir:,}"],
            ["Self-loops", f"{self_loops:,}"],
            ["Isolated nodes", f"{isolated:,}"],
            ["Density (undirected)", f"{density:.6f}"],
            ["Sparsity (1-density)", f"{sparsity:.6f}"],
            ["x_dim", str(x_dim)],
            ["issuer_index", "yes" if issuer_index is not None else "no"],
            ["corr_edge_mask", "yes" if corr_mask is not None else "no"],
        ]
        ov_figs = _make_table_pages(overview_headers, overview_rows, title="PyG dataset overview", rows_per_page=26)
        for f in reversed(ov_figs):
            figures.insert(0, ("Overview (table)", f))

        # Schema tables
        schema_headers = ["Field", "Value"]
        schema_rows = [["x_dim", str(x_dim)]]
        schema_rows.append(["issuer_index", ("yes" if issuer_index is not None else "no")])
        if meta and isinstance(meta.get("feature_dims"), dict):
            fd = meta["feature_dims"]
            for k in ["num_nodes", "num_edges_directed", "num_relations"]:
                if k in fd:
                    schema_rows.append([k, str(fd[k])])
        # Relations list table (id, name)
        if rel_names:
            rel_headers = ["Relation #", "Name"]
            rel_rows = [[str(i), str(n)] for i, n in enumerate(rel_names)]
            rel_figs = _make_table_pages(rel_headers, rel_rows, title="Relations", rows_per_page=30)
        else:
            rel_figs = []
        # Feature columns table
        feat_figs = []
        if meta and isinstance(meta.get("node_feature_schema"), dict):
            cols = meta["node_feature_schema"].get("columns")
            if isinstance(cols, list):
                feat_headers = ["#", "Feature column"]
                feat_rows = [[str(i), str(c)] for i, c in enumerate(cols)]
                feat_figs = _make_table_pages(feat_headers, feat_rows, title="Feature columns", rows_per_page=34)
        # Combine: first a small schema summary page
        schema_summary_figs = _make_table_pages(schema_headers, schema_rows, title="Feature schema summary", rows_per_page=26)
        # Insert in order: schema summary, relations, feature columns
        insertion = []
        insertion.extend([("Feature schema (summary)", f) for f in schema_summary_figs])
        insertion.extend([("Relations", f) for f in rel_figs])
        insertion.extend([("Feature columns", f) for f in feat_figs])
        for f_title, f in reversed(insertion):
            figures.insert(1, (f_title, f))  # after overview
    except Exception as e:
        print(f"[yellow]Failed to build table pages: {e}[/yellow]")
        print("[hint] Falling back to text pages in PDF.")

    # PDF report
    if pdf:
        pdf_path = _write_pdf(
            outdir,
            stem=pt_path.stem,
            overview_lines=None,  # we already added a table page
            figures=figures,
            metrics_lines=metrics,
            hints_lines=hints,
            schema_lines=None,  # schema covered by table pages
        )
        if pdf_path:
            print(f"[green]PDF report written:[/green] {pdf_path}")

    # Also save individual figures as PNGs
    try:
        import matplotlib.pyplot as plt  # type: ignore
        outdir.mkdir(parents=True, exist_ok=True)
        for title, fig in figures:
            safe = title.lower().replace(" ", "_").replace("/", "-")
            p = outdir / f"{pt_path.stem}__{safe}.png"
            fig.savefig(p, dpi=130)
            try:
                plt.close(fig)
            except Exception:
                pass
        print(f"[OK] Figures saved to {outdir}")
    except Exception as e:
        print(f"[yellow]Failed to save PNG figures: {e}[/yellow]")

    # Final safety: ensure no figures remain open
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.close('all')
    except Exception:
        pass

    return 0


# --- helper: feature histogram pages ---
def _feature_hist_pages(x: np.ndarray, names: List[str], max_cols: int = 24, bins: int = 40, per_page: int = 6):
    import matplotlib.pyplot as plt  # type: ignore
    figs = []
    C = x.shape[1]
    sel = min(C, max_cols)
    cols = list(range(sel))
    # Simple heuristics for usefulness flags
    eps_std = 1e-6
    for i0 in range(0, sel, per_page):
        i1 = min(sel, i0 + per_page)
        k = i1 - i0
        nrows, ncols = 3, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, 11))
        axes = axes.flatten()
        for j, ci in enumerate(cols[i0:i1]):
            ax = axes[j]
            col = x[:, ci]
            finite = np.isfinite(col)
            col = col[finite]
            name = names[ci] if ci < len(names) else f"f{ci}"
            if col.size == 0:
                ax.text(0.5, 0.5, f"{name}\n(no finite values)", ha='center', va='center')
                ax.axis('off')
                continue
            mu = float(np.nanmean(col))
            sd = float(np.nanstd(col))
            pct_zero = float((np.abs(col) < 1e-12).mean() * 100.0)
            ax.hist(col, bins=bins, color="#4C72B0", edgecolor="white")
            ax.set_title(f"{name}\nμ={mu:.3g} σ={sd:.3g} zero%={pct_zero:.1f}", fontsize=9)
        # turn off any unused axes
        for j in range(k, len(axes)):
            axes[j].axis('off')
        fig.tight_layout()
        figs.append(fig)
    # Build a summary page highlighting near-constant features
    try:
        near_const = []
        for ci in range(sel):
            col = x[:, ci]
            col = col[np.isfinite(col)]
            if col.size == 0:
                near_const.append((ci, names[ci] if ci < len(names) else f"f{ci}", "no finite"))
                continue
            sd = float(np.nanstd(col))
            if sd < eps_std:
                near_const.append((ci, names[ci] if ci < len(names) else f"f{ci}", f"σ={sd:.2e}"))
        if near_const:
            fig, ax = plt.subplots(figsize=(8.5, 3 + 0.25*len(near_const)))
            ax.axis('off')
            headers = ["#", "Feature", "Flag"]
            rows = [[str(i), n, flag] for (i, n, flag) in near_const]
            table = ax.table(cellText=rows, colLabels=headers, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.2)
            ax.set_title("Features flagged as near-constant (may be uninformative)")
            fig.tight_layout()
            figs.insert(0, fig)
    except Exception:
        pass
    return figs
