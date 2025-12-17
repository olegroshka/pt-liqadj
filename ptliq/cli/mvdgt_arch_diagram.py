from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import typer


app = typer.Typer(no_args_is_help=True)


def _load_meta(workdir: Path) -> dict:
    meta_path = workdir / "mvdgt_meta.json"
    if not meta_path.exists():
        raise typer.BadParameter(f"mvdgt_meta.json not found in workdir: {workdir}")
    try:
        return json.loads(meta_path.read_text())
    except Exception as e:
        raise typer.BadParameter(f"Failed to read {meta_path}: {e}")


def _load_model_cfg(workdir: Path) -> dict:
    cfg_path = workdir / "model_config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text())
    except Exception:
        return {}


def _bool_from_cfg(cfg: dict, key: str, default: bool) -> bool:
    v = cfg.get(key, default)
    try:
        return bool(v)
    except Exception:
        return bool(default)


def _detect_views(cfg: dict) -> list[str]:
    vs = cfg.get("views")
    if isinstance(vs, list) and all(isinstance(x, str) for x in vs) and len(vs) > 0:
        return list(vs)
    return ["struct", "port", "corr_global", "corr_local"]


def _require_graphviz():
    try:
        import graphviz  # noqa: F401
    except Exception as e:
        raise typer.BadParameter(
            "Graphviz Python package is not available.\n"
            "Please install both:\n"
            "  1) Graphviz system binary: https://graphviz.org/download/\n"
            "  2) Python package: pip install graphviz\n\n"
            f"Import error: {e}"
        )


def _apply_preset(
    preset: Optional[str],
    fmt: str,
    width: Optional[float],
    height: Optional[float],
    dpi: Optional[int],
    rankdir: Optional[str],
    square: bool,
):
    """Apply named layout presets. Explicit CLI values always override presets."""
    if not preset:
        return width, height, dpi, rankdir, square

    p = str(preset).lower()
    if p == "ppt4x3":
        width = width if width is not None else 12.0
        height = height if height is not None else 9.0
        if dpi is None and fmt in {"png", "jpg", "jpeg"}:
            dpi = 300
        # Prefer top→bottom for slides unless user explicitly set --rankdir
        rankdir = (rankdir or "TB")
        square = False
    elif p == "ppt16x9":
        width = width if width is not None else 12.0
        height = height if height is not None else 6.75
        if dpi is None and fmt in {"png", "jpg", "jpeg"}:
            dpi = 300
        rankdir = (rankdir or "LR")
        square = False
    elif p == "square":
        width = width if width is not None else 8.0
        height = height if height is not None else 8.0
        if dpi is None and fmt in {"png", "jpg", "jpeg"}:
            dpi = 300
        rankdir = (rankdir or "TB")
        square = True
    # Unknown preset → no changes
    return width, height, dpi, rankdir, square


def _make_diagram(
    out_stem: str,
    view_names: list[str],
    use_market: bool,
    use_trade: bool,
    use_portfolio_attn: bool,
    portfolio_attn_mode: str,
    use_pf_head: bool,
    show_negative_drag: bool,
    format: str,
    # layout controls
    rankdir: str = "LR",
    square: bool = False,
    width: Optional[float] = None,
    height: Optional[float] = None,
    dpi: Optional[int] = None,
    compact: bool = False,
    include_notes: bool = False,
) -> Path:
    from graphviz import Digraph

    # graph attributes
    graph_attr = {
        "rankdir": (rankdir or "LR").upper(),
        "splines": "spline",
        "fontsize": "14",
        "fontname": "Segoe UI, Roboto, Arial",
        "labelloc": "t",
        "pad": "0.25",
        "newrank": "true",
    }
    # spacings
    if compact:
        graph_attr.update({"nodesep": "0.38", "ranksep": "0.55"})
    else:
        graph_attr.update({"nodesep": "0.65", "ranksep": "0.9"})

    # size/aspect controls (inches). When set, Graphviz respects it for raster exports
    if square and (width is None and height is None):
        width = height = 8.0
    if width is not None or height is not None:
        w = max(0.1, float(width if width is not None else (height or 8.0)))
        h = max(0.1, float(height if height is not None else (width or 8.0)))
        # '!' forces exact size and aspect
        graph_attr["size"] = f"{w},{h}!"
    # Ensure readable raster by default when exporting to bitmap
    if dpi is None and format.lower() in {"png", "jpg", "jpeg"}:
        dpi = 300
    if dpi is not None:
        graph_attr["dpi"] = str(int(max(96, dpi)))

    g = Digraph(
        "MV_DGT",
        graph_attr=graph_attr,
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "color": "#CBD5E1",
            "fillcolor": "#F8FAFC",
            "fontname": "Segoe UI, Roboto, Arial",
            "fontsize": "14",
            "margin": "0.16,0.11",
        },
        edge_attr={"color": "#64748B"},
    )

    # --- Multi‑view Graph Encoder ---
    with g.subgraph(name="cluster_graph") as c:
        c.attr(label="Multi‑View Graph Encoder (2 TransformerConv layers)", color="#4C78A8")
        c.node(
            "NodeEnc",
            "Node Feature Encoder\n"
            "x∈R^{N×F} → H∈R^{N×D} via 2‑layer MLP + ReLU + Dropout + LayerNorm",
        )

        # Views present
        has_struct = "struct" in view_names
        has_port = "port" in view_names
        has_cg = "corr_global" in view_names
        has_cl = "corr_local" in view_names

        if has_struct:
            c.node("Struct", "Graph Convolution — Structural Edges\n(TransformerConv)")
        if has_port:
            c.node("Port", "Graph Convolution — Portfolio Co‑membership Edges\n(TransformerConv)")
        if has_cg:
            c.node("CorrG", "Graph Convolution — Global Correlation Edges\n(TransformerConv, gated)")
        if has_cl:
            c.node("CorrL", "Graph Convolution — Local Correlation Edges\n(TransformerConv, gated)")

        c.node(
            "DiffFuse",
            "Differential Fusion with Learnable Gates\n"
            "(Structural baseline + per‑view deviations)",
        )

        # Edges from NodeEnc to each view, then to fusion
        if has_struct:
            c.edge("NodeEnc", "Struct")
            c.edge("Struct", "DiffFuse")
        if has_port:
            c.edge("NodeEnc", "Port")
            c.edge("Port", "DiffFuse")
        if has_cg:
            c.edge("NodeEnc", "CorrG")
            c.edge("CorrG", "DiffFuse")
        if has_cl:
            c.edge("NodeEnc", "CorrL")
            c.edge("CorrL", "DiffFuse")

    g.node(
        "Anchor",
        "Anchor Selection\n"
        "z_anchor_pre ∈ R^{B×D} = H[anchor_idx]",
    )
    g.edge("DiffFuse", "Anchor", minlen="2")

    # --- Portfolio (strict LOO) ---
    with g.subgraph(name="cluster_pf") as c:
        c.attr(label="Portfolio Prototype (Strict Leave‑One‑Out)", color="#F58518")
        c.node(
            "Vabs",
            "Absolute Portfolio Prototype V_abs ∈ R^{B×D}\n"
            "strict LOO in H; L2‑normalized",
        )
        c.node("PFProj", "Projection pf_proj: R^{2D}→R^{D}\n([V_abs, 0] input)")
        c.node("PFGate", "Residual gate σ(pf_gate) ∈ (0,1)")
        c.edge("Vabs", "PFProj")
        c.edge("PFProj", "PFGate")

    g.node(
        "ZAnchor",
        "Portfolio Residual Fusion\n"
        "z_anchor = z_anchor_pre − σ(pf_gate)·pf_proj([V_abs, 0]) ∈ R^{B×D}",
    )
    g.edge("Anchor", "ZAnchor")
    g.edge("PFGate", "ZAnchor")

    # --- Context encoders ---
    if use_market or use_trade:
        with g.subgraph(name="cluster_ctx") as c:
            c.attr(label="Sample‑Level Context Encoders", color="#54A24B")
            if use_market:
                c.node("Mkt", "Market Context Encoder\n"
                       "mkt_enc: R^{m}→R^{D}")
            if use_trade:
                c.node("Trade", "Trade / Microstructure Encoder\n"
                       "trade_enc: R^{t}→R^{D}")

    # --- Optional within‑portfolio self‑attention ---
    if use_portfolio_attn:
        attn_label = (
            "Within‑Portfolio Self‑Attention (optional)\n"
            "TransformerEncoder over portfolio tokens: R^{B×D}→R^{B×D}"
        )
        g.node("Attn", attn_label)
        g.edge("ZAnchor", "Attn")
        if use_market:
            g.edge("Mkt", "Attn")
        if use_trade:
            g.edge("Trade", "Attn")

    # --- Optional portfolio head (V_abs head) ---
    if use_pf_head:
        g.node("PFHead", "Portfolio Head (optional)\nMLP(V_abs) → R^{D} (gated)")
        g.edge("Vabs", "PFHead")

    # --- Regression head ---
    g.node("Head", "Regression Head\nconcat(·) → MLP → ŷ ∈ R^{B}")

    # Inputs to head (set constraint=false to avoid stretching ranks)
    g.edge("ZAnchor", "Head", constraint="false")
    if use_market:
        g.edge("Mkt", "Head", constraint="false")
    if use_trade:
        g.edge("Trade", "Head", constraint="false")
    if use_portfolio_attn and portfolio_attn_mode == "concat":
        g.edge("Attn", "Head", constraint="false")
    if use_pf_head:
        g.edge("PFHead", "Head", constraint="false")

    # --- Negative drag annotation ---
    if show_negative_drag:
        g.node(
            "NegDrag",
            "Deterministic Negative Drag (optional)\n"
            "ŷ ← ŷ − c·|cos( normalize(z_anchor_pre), V_abs )|",
        )
        g.edge("Anchor", "NegDrag", style="dashed")
        g.edge("Vabs", "NegDrag", style="dashed")
        g.edge("NegDrag", "Head", style="dashed")

    # --- Middle layout ---
    # For square canvases, arrange the middle into two rows to reduce width.
    if square:
        row1: list[str] = ["ZAnchor"]
        if use_market:
            row1.append("Mkt")
        if use_trade:
            row1.append("Trade")
        row2: list[str] = []
        if use_pf_head:
            row2.append("PFHead")
        if use_portfolio_attn:
            row2.append("Attn")
        row2.append("Head")
        if show_negative_drag:
            row2.append("NegDrag")

        if row1:
            with g.subgraph(name="rank_mid1") as r1:
                r1.attr(rank="same")
                for n in row1:
                    r1.node(n)
            # stabilize ordering
            for a, b in zip(row1, row1[1:]):
                g.edge(a, b, style="invis")
        if row2:
            with g.subgraph(name="rank_mid2") as r2:
                r2.attr(rank="same")
                for n in row2:
                    r2.node(n)
            for a, b in zip(row2, row2[1:]):
                g.edge(a, b, style="invis")
        # Encourage vertical alignment between rows
        g.edge(row1[0], row2[0], style="invis")
    else:
        # Single horizontal middle row
        mid_nodes: list[str] = ["ZAnchor", "Head"]
        if use_market:
            mid_nodes.append("Mkt")
        if use_trade:
            mid_nodes.append("Trade")
        if use_portfolio_attn:
            mid_nodes.append("Attn")
        if use_pf_head:
            mid_nodes.append("PFHead")
        if show_negative_drag:
            mid_nodes.append("NegDrag")

        if mid_nodes:
            with g.subgraph(name="rank_mid") as r:
                r.attr(rank="same")
                for n in mid_nodes:
                    r.node(n)
            # Create an invisible chain to stabilize left→right ordering
            chain = [n for n in [
                "ZAnchor",
                "Mkt" if use_market else None,
                "Trade" if use_trade else None,
                "Attn" if use_portfolio_attn else None,
                "PFHead" if use_pf_head else None,
                "Head",
                "NegDrag" if show_negative_drag else None,
            ] if n is not None]
            for a, b in zip(chain, chain[1:]):
                g.edge(a, b, style="invis")

    # --- Top rank enforcement (Graph Encoder must be at the top in TB layout) ---
    with g.subgraph(name="rank_top") as rtop:
        rtop.attr(rank="source")
        # List explicit nodes that belong to the top layer
        rtop.node("NodeEnc")
        if "struct" in view_names:
            rtop.node("Struct")
        if "port" in view_names:
            rtop.node("Port")
        if "corr_global" in view_names:
            rtop.node("CorrG")
        if "corr_local" in view_names:
            rtop.node("CorrL")
        rtop.node("DiffFuse")

    # --- Explanatory notes (bottom) ---
    if include_notes:
        notes_label = (
            "<"
            "<b>Notes</b><BR ALIGN=\"LEFT\"/>"
            "• Anchor: focal asset representation gathered from the Multi‑View Graph Encoder; serves as the per‑sample target for portfolio residual fusion.<BR ALIGN=\"LEFT\"/>"
            "• Multi‑view fusion: structural edges act as a baseline; portfolio and correlation views contribute only their <i>deviations</i> from structural via learnable gates (differential fusion).<BR ALIGN=\"LEFT\"/>"
            "• Strict LOO prototype V_abs: computed for the anchor’s portfolio group without the anchor itself; absolute‑weighted average in embedding space H; L2‑normalized to compare directions reliably.<BR ALIGN=\"LEFT\"/>"
            "• Residual subtraction: z_anchor = z_anchor_pre − σ(pf_gate)·pf_proj([V_abs, 0]); removes shared portfolio drift while preserving anchor‑specific information.<BR ALIGN=\"LEFT\"/>"
            "• Market / Trade encoders: add exogenous, sample‑level context (macro regime, microstructure). These are small MLPs projected to the shared hidden space.<BR ALIGN=\"LEFT\"/>"
            "• Within‑portfolio self‑attention (optional): encodes interactions among items in the same portfolio batch group; used in residual or concat mode controlled by a learnable gate.<BR ALIGN=\"LEFT\"/>"
            "• Portfolio head (optional): clean branch over V_abs to expose portfolio prototype features directly to the prediction head (gated).<BR ALIGN=\"LEFT\"/>"
            "• Deterministic negative drag (optional): subtracts |cos(z_anchor_pre, V_abs)| scaled by a coefficient to bias outputs away from portfolio direction, improving conservativeness.<BR ALIGN=\"LEFT\"/>"
            "• Training: standard regression head on concatenated features; typical objective is MSE/Huber with regularization; graph and portfolio paths trained end‑to‑end.<BR ALIGN=\"LEFT\"/>"
            "• Typical shapes: x∈R^{N×F}, anchor_idx∈N^{B}, market_feat∈R^{B×m}, trade_feat∈R^{B×t}; hidden size H (shared across blocks).<BR ALIGN=\"LEFT\"/>"
            "• Inference path: encode graph → gather anchor → subtract portfolio residual → add optional context/attention → predict via head; negative drag applied if enabled.<BR ALIGN=\"LEFT\"/>"
            ">"
        )
        with g.subgraph(name="cluster_notes") as c:
            c.attr(label="", color="#94A3B8")
            c.node(
                "Notes",
                notes_label,
                shape="box",
                style="rounded,filled",
                fillcolor="#FFFFFF",
                color="#CBD5E1",
            )
        # Encourage notes to sit at the bottom and span width
        for n in ["ZAnchor", "Head", "Mkt" if use_market else None, "Trade" if use_trade else None, "Attn" if use_portfolio_attn else None, "PFHead" if use_pf_head else None]:
            if n is not None:
                g.edge(n, "Notes", style="invis")
        # Pin Notes to the bottom rank in TB layout
        with g.subgraph(name="rank_notes") as rn:
            rn.attr(rank="sink")
            rn.node("Notes")

    # Always emit a DOT alongside the rendered artifact for portability
    dot_path = Path(out_stem + ".dot")
    try:
        g.save(filename=str(dot_path))
    except Exception:
        # Saving DOT rarely fails, continue to try rendering
        pass

    # Try to render with Graphviz system binary; gracefully fall back to DOT
    try:
        out_path = Path(g.render(out_stem, format=format, cleanup=True))
        return out_path
    except Exception as e:
        try:
            # Detect common 'dot' missing error class without importing internals explicitly
            import graphviz as _gv  # type: ignore
            from graphviz.backend import ExecutableNotFound as _ExecNF  # type: ignore
            is_exec_missing = isinstance(e, _ExecNF)
        except Exception:
            is_exec_missing = False

        # Fall back to DOT file when system Graphviz is not available or any render error occurs
        fallback_msg = (
            f"[warn] Failed to render diagram to {format}: {e}.\n"
            f"Saved Graphviz DOT to: {dot_path}.\n"
            "Install Graphviz system binary to enable SVG/PNG/PDF rendering: https://graphviz.org/download/"
        )
        typer.echo(fallback_msg)
        return dot_path


@app.callback(invoke_without_command=True)
def main(
    workdir: Path = typer.Option(
        Path("data/mvdgt/dgt_default"), help="MV‑DGT working directory (expects mvdgt_meta.json, model_config.json)"
    ),
    out: Optional[Path] = typer.Option(
        None,
        help="Output diagram path; extension decides format (svg/png/pdf). Default: <workdir>/mv_dgt_concept.svg",
    ),
    preset: Optional[str] = typer.Option(
        None,
        help="Convenience layout preset: ppt4x3 | ppt16x9 | square. Explicit width/height/dpi override the preset.",
    ),
    format: Optional[str] = typer.Option(
        None,
        help="Diagram format: svg | png | pdf. If not set, derived from output extension or defaults to svg.",
    ),
    force_market: Optional[bool] = typer.Option(
        None,
        help="Override: include Market Context Encoder (default inferred from model_config).",
    ),
    force_trade: Optional[bool] = typer.Option(
        None,
        help="Override: include Trade/Microstructure Encoder (default inferred from model_config).",
    ),
    force_portfolio_attn: Optional[bool] = typer.Option(
        None,
        help="Override: include Within‑Portfolio Self‑Attention block (default inferred from model_config).",
    ),
    force_pf_head: Optional[bool] = typer.Option(
        None,
        help="Override: include Portfolio Head on V_abs (default inferred from model_config).",
    ),
    show_negative_drag: bool = typer.Option(
        True, help="Annotate the deterministic negative‑drag mechanism on the diagram."
    ),
    # layout & presentation options
    rankdir: Optional[str] = typer.Option(
        None, help="Graph layout direction: LR (left→right) or TB (top→bottom). If omitted, preset decides."
    ),
    square: bool = typer.Option(
        False, help="Force a square canvas if width/height are not provided (uses 8×8 inches)."
    ),
    width: Optional[float] = typer.Option(
        None, help="Target diagram width in inches (Graphviz 'size' attribute)."
    ),
    height: Optional[float] = typer.Option(
        None, help="Target diagram height in inches (Graphviz 'size' attribute)."
    ),
    dpi: Optional[int] = typer.Option(
        None, help="Raster DPI for PNG/JPG export (Graphviz 'dpi' graph attribute)."
    ),
    compact: bool = typer.Option(
        True, help="Use tighter node/rank spacing to better fit slides."
    ),
    include_notes: bool = typer.Option(
        True, help="Add a notes panel at the bottom with short explanations of key components."
    ),
):
    """Produce a clean, presentation‑quality MV‑DGT architecture diagram using Graphviz.

    The diagram intentionally abstracts away low‑level operators and shows the conceptual
    blocks: multi‑view graph encoder, portfolio LOO prototype and residual fusion, optional
    sample‑level context encoders, optional within‑portfolio self‑attention, portfolio head,
    and the final regression head with optional negative drag.
    """
    workdir = Path(workdir)
    if not workdir.exists():
        raise typer.BadParameter(f"workdir does not exist: {workdir}")

    # Read basic config to infer which blocks to show
    _ = _load_meta(workdir)  # validate presence; actual contents not required for diagram
    cfg = _load_model_cfg(workdir)

    view_names = _detect_views(cfg)
    use_market = _bool_from_cfg(cfg, "use_market", True)
    use_trade = bool(int(cfg.get("trade_dim", 0)) > 0)
    use_portfolio_attn = _bool_from_cfg(cfg, "use_portfolio_attn", False)
    portfolio_attn_mode = str(cfg.get("portfolio_attn_mode", "residual"))
    use_pf_head = _bool_from_cfg(cfg, "use_pf_head", False)

    if force_market is not None:
        use_market = bool(force_market)
    if force_trade is not None:
        use_trade = bool(force_trade)
    if force_portfolio_attn is not None:
        use_portfolio_attn = bool(force_portfolio_attn)
    if force_pf_head is not None:
        use_pf_head = bool(force_pf_head)

    # Decide output path and format
    if out is None:
        out = workdir / "mv_dgt_concept.svg"
    out = Path(out)
    fmt = (format or out.suffix.lstrip(".") or "svg").lower()
    if fmt not in {"svg", "png", "pdf", "dot"}:
        raise typer.BadParameter(f"Unsupported format: {fmt}. Use svg | png | pdf | dot.")

    # Ensure Python graphviz is importable (needed at least to emit DOT)
    _require_graphviz()

    # Apply preset defaults (without overwriting explicit flags)
    width2, height2, dpi2, rankdir2, square2 = _apply_preset(
        preset=preset, fmt=fmt, width=width, height=height, dpi=dpi, rankdir=rankdir, square=square
    )

    # Graphviz render() takes a stem without extension
    out_stem = str(out.with_suffix(""))

    path = _make_diagram(
        out_stem=out_stem,
        view_names=view_names,
        use_market=use_market,
        use_trade=use_trade,
        use_portfolio_attn=use_portfolio_attn,
        portfolio_attn_mode=portfolio_attn_mode,
        use_pf_head=use_pf_head,
        show_negative_drag=bool(show_negative_drag),
        format=fmt,
        rankdir=rankdir2,
        square=square2,
        width=width2,
        height=height2,
        dpi=dpi2,
        compact=compact,
        include_notes=include_notes,
    )

    typer.echo(f"[ok] MV‑DGT architecture diagram saved to: {path}")


if __name__ == "__main__":
    app()
