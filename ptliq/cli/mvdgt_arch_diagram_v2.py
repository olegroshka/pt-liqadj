
from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import typer

app = typer.Typer(no_args_is_help=True)


# -----------------------------
# I/O helpers
# -----------------------------
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


# -----------------------------
# Layout presets (slide-friendly)
# -----------------------------
def _apply_preset(
    preset: Optional[str],
    fmt: str,
    width: Optional[float],
    height: Optional[float],
    dpi: Optional[int],
    rankdir: Optional[str],
    square: bool,
    notes: Optional[str],
):
    """
    Apply named layout presets. Explicit CLI values override presets.
    Notes:
      - For slides, "TB" reads best and avoids the ultra-wide sprawl.
      - Prefer SVG/PDF for crisp scaling in Keynote/PowerPoint.
    """
    if not preset:
        return width, height, dpi, rankdir, square, notes

    p = str(preset).lower()
    if p == "ppt4x3":
        width = width if width is not None else 10.5
        height = height if height is not None else 7.5
        if dpi is None and fmt in {"png", "jpg", "jpeg"}:
            dpi = 300
        rankdir = (rankdir or "TB")
        square = False
        notes = notes or "short"
    elif p == "ppt16x9":
        width = width if width is not None else 12.5
        height = height if height is not None else 7.0
        if dpi is None and fmt in {"png", "jpg", "jpeg"}:
            dpi = 300
        rankdir = (rankdir or "TB")
        square = False
        notes = notes or "short"
    elif p == "square":
        width = width if width is not None else 9.0
        height = height if height is not None else 9.0
        if dpi is None and fmt in {"png", "jpg", "jpeg"}:
            dpi = 300
        rankdir = (rankdir or "TB")
        square = True
        notes = notes or "short"

    return width, height, dpi, rankdir, square, notes


# -----------------------------
# Label helpers (clean + consistent)
# -----------------------------
def _html_box(title: str, body: str, dims: str | None = None) -> str:
    """
    Create a compact HTML-like label for Graphviz.
    Avoid long single lines (they blow up node width).
    """
    if dims:
        dims_row = f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#64748B">{dims}</FONT></TD></TR>'
    else:
        dims_row = ""

    # Keep body short; use <BR/> for line breaks.
    return (
        "<"
        "<TABLE BORDER='0' CELLBORDER='0' CELLPADDING='2'>"
        f"<TR><TD ALIGN='LEFT'><B>{title}</B></TD></TR>"
        f"<TR><TD ALIGN='LEFT'>{body}</TD></TR>"
        f"{dims_row}"
        "</TABLE>"
        ">"
    )


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
    rankdir: str = "TB",
    square: bool = False,
    width: Optional[float] = None,
    height: Optional[float] = None,
    dpi: Optional[int] = None,
    compact: bool = True,
    notes: str = "short",  # none | short | full
) -> Path:
    from graphviz import Digraph

    rankdir = (rankdir or "TB").upper()
    notes = (notes or "short").lower()

    # Graph attributes: tuned for slide readability and compactness
    graph_attr = {
        "rankdir": rankdir,
        "splines": "ortho",         # cleaner elbows; tends to reduce spaghetti
        "concentrate": "true",
        "newrank": "true",
        "bgcolor": "white",
        "fontsize": "14",
        "fontname": "Segoe UI, Roboto, Arial",
        "pad": "0.18",
        "nodesep": "0.25" if compact else "0.45",
        "ranksep": "0.40" if compact else "0.70",
    }

    # Size/aspect controls (inches). Use ratio=compress to avoid ultra-wide outputs.
    if square and (width is None and height is None):
        width = height = 9.0
    if width is not None or height is not None:
        w = max(0.1, float(width if width is not None else (height or 9.0)))
        h = max(0.1, float(height if height is not None else (width or 9.0)))
        graph_attr["size"] = f"{w},{h}"
        graph_attr["ratio"] = "compress"

    # Raster DPI (SVG/PDF ignore this)
    if dpi is None and format.lower() in {"png", "jpg", "jpeg"}:
        dpi = 300
    if dpi is not None:
        graph_attr["dpi"] = str(int(max(120, dpi)))

    g = Digraph(
        "MV_DGT",
        graph_attr=graph_attr,
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "color": "#CBD5E1",
            "fillcolor": "#F8FAFC",
            "fontname": "Segoe UI, Roboto, Arial",
            "fontsize": "13",
            "margin": "0.12,0.08",
        },
        edge_attr={"color": "#64748B", "arrowsize": "0.7"},
    )

    # -----------------------------
    # Multi‑view Graph Encoder
    # -----------------------------
    with g.subgraph(name="cluster_graph") as c:
        c.attr(
            label="Multi‑View Graph Encoder (2 layers)",
            color="#4C78A8",
            style="rounded",
            penwidth="1.3",
        )

        c.node(
            "NodeEnc",
            _html_box(
                "Node Encoder",
                "2‑layer MLP + LayerNorm",
                dims="x: [N, F] → H: [N, D]",
            ),
            shape="plain",
        )

        has_struct = "struct" in view_names
        has_port = "port" in view_names
        has_cg = "corr_global" in view_names
        has_cl = "corr_local" in view_names

        # Per-view conv nodes
        if has_struct:
            c.node(
                "Struct",
                _html_box("View: Structural", "TransformerConv", dims="edges: mask_struct"),
                shape="plain",
            )
        if has_port:
            c.node(
                "Port",
                _html_box("View: Co‑portfolio", "TransformerConv", dims="edges: mask_port"),
                shape="plain",
            )
        if has_cg:
            c.node(
                "CorrG",
                _html_box("View: Corr (global)", "TransformerConv + corr_gate", dims="edges: mask_corr_global"),
                shape="plain",
            )
        if has_cl:
            c.node(
                "CorrL",
                _html_box("View: Corr (local)", "TransformerConv + corr_gate", dims="edges: mask_corr_local"),
                shape="plain",
            )

        c.node(
            "DiffFuse",
            _html_box(
                "Differential Fusion",
                "Structural baseline + gated deviations",
                dims="per layer: learn σ(g[v])",
            ),
            shape="plain",
        )

        # --- Layout inside the cluster (prevents the "super-wide top row") ---
        # Row 0: NodeEnc
        with c.subgraph(name="rank_graph_0") as r0:
            r0.attr(rank="same")
            r0.node("NodeEnc")

        # Row 1: two view nodes (if present)
        row1 = [n for n in ["Struct", "Port"] if (n in {"Struct", "Port"} and ((n == "Struct" and has_struct) or (n == "Port" and has_port)))]
        # Row 2: remaining two view nodes (if present)
        row2 = [n for n in ["CorrG", "CorrL"] if (n in {"CorrG", "CorrL"} and ((n == "CorrG" and has_cg) or (n == "CorrL" and has_cl)))]

        if row1:
            with c.subgraph(name="rank_graph_1") as r1:
                r1.attr(rank="same")
                for n in row1:
                    r1.node(n)
            for a, b in zip(row1, row1[1:]):
                c.edge(a, b, style="invis")

        if row2:
            with c.subgraph(name="rank_graph_2") as r2:
                r2.attr(rank="same")
                for n in row2:
                    r2.node(n)
            for a, b in zip(row2, row2[1:]):
                c.edge(a, b, style="invis")

        # Row 3: fusion
        with c.subgraph(name="rank_graph_3") as r3:
            r3.attr(rank="same")
            r3.node("DiffFuse")

        # Edges: NodeEnc → each view → fusion
        for n in ["Struct", "Port", "CorrG", "CorrL"]:
            if (n == "Struct" and has_struct) or (n == "Port" and has_port) or (n == "CorrG" and has_cg) or (n == "CorrL" and has_cl):
                c.edge("NodeEnc", n)
                c.edge(n, "DiffFuse")

    # -----------------------------
    # Anchor gather
    # -----------------------------
    g.node(
        "Anchor",
        _html_box("Gather Anchor", "Select focal node embeddings", dims="z_pre = H[anchor_idx] → [B, D]"),
        shape="plain",
    )
    g.edge("DiffFuse", "Anchor")

    # -----------------------------
    # Portfolio (strict LOO) prototype + residual subtraction
    # -----------------------------
    with g.subgraph(name="cluster_pf") as c:
        c.attr(label="Portfolio De‑bias (strict LOO)", color="#F58518", style="rounded", penwidth="1.3")
        c.node(
            "Vabs",
            _html_box("LOO Prototype", "Absolute‑weighted, L2‑norm", dims="V_abs: [B, D] (excludes anchor)"),
            shape="plain",
        )
        c.node(
            "PFGate",
            _html_box("Residual Gate", "σ(pf_gate) ∈ (0,1)", dims="scalar (learned)"),
            shape="plain",
        )
        c.node(
            "PFProj",
            _html_box("Projection", "pf_proj([V_abs, 0])", dims="[2D] → [D]"),
            shape="plain",
        )
        c.edge("Vabs", "PFProj")
        c.edge("PFProj", "PFGate")

    g.node(
        "ZAnchor",
        _html_box("Residual Anchor", "z = z_pre − gate · proj(V_abs)", dims="z_anchor: [B, D]"),
        shape="plain",
    )
    g.edge("Anchor", "ZAnchor")
    # Light cue that the portfolio prototype is computed per-anchor sample (via pf_gid / port_ctx)
    g.edge("Anchor", "Vabs", style="dashed", constraint="true")
    g.edge("PFGate", "ZAnchor")

    # -----------------------------
    # Optional sample-level context
    # -----------------------------
    ctx_nodes: list[str] = []
    if use_market or use_trade:
        with g.subgraph(name="cluster_ctx") as c:
            c.attr(label="Sample‑Level Context (optional)", color="#54A24B", style="rounded", penwidth="1.3")
            if use_market:
                c.node("Mkt", _html_box("Market Encoder", "small MLP → D", dims="market_feat: [B, m]"), shape="plain")
                ctx_nodes.append("Mkt")
            if use_trade:
                c.node("Trade", _html_box("Trade Encoder", "small MLP → D", dims="trade_feat: [B, t]"), shape="plain")
                ctx_nodes.append("Trade")

    # -----------------------------
    # Optional within-portfolio attention
    # -----------------------------
    if use_portfolio_attn:
        mode = "residual" if portfolio_attn_mode == "residual" else "concat"
        g.node(
            "Attn",
            _html_box("Within‑Portfolio Attention", f"TransformerEncoder (mode: {mode})", dims="tokens grouped by pf_gid"),
            shape="plain",
        )
        g.edge("ZAnchor", "Attn")
        if use_market:
            g.edge("Mkt", "Attn")
        if use_trade:
            g.edge("Trade", "Attn")

    # -----------------------------
    # Optional portfolio head branch
    # -----------------------------
    if use_pf_head:
        g.node("PFHead", _html_box("Portfolio Head", "MLP(V_abs) (gated)", dims="→ [B, D]"), shape="plain")
        g.edge("Vabs", "PFHead")

    # -----------------------------
    # Head + output
    # -----------------------------
    g.node(
        "Head",
        _html_box("Regression Head", "concat(selected branches) → MLP → ŷ", dims="ŷ: [B]"),
        shape="plain",
    )

    # To avoid stretching the whole layout, route everything into a tiny "concat" junction.
    g.node("J", "", shape="point", width="0.04", height="0.04")

    g.edge("ZAnchor", "J")
    if use_market:
        g.edge("Mkt", "J")
    if use_trade:
        g.edge("Trade", "J")
    if use_portfolio_attn and portfolio_attn_mode == "concat":
        g.edge("Attn", "J")
    if use_pf_head:
        g.edge("PFHead", "J")

    g.edge("J", "Head")

    # Negative drag: drawn as a dashed side note, not a main-path block
    if show_negative_drag:
        g.node(
            "NegDrag",
            _html_box("Negative Drag (optional)", "− c · |cos(z_pre, V_abs)|", dims="signless, strict‑LOO"),
            shape="plain",
        )
        g.edge("Anchor", "NegDrag", style="dashed")
        g.edge("Vabs", "NegDrag", style="dashed")
        g.edge("NegDrag", "Head", style="dashed")

    # -----------------------------
    # Notes / legend (bottom)
    # -----------------------------
    if notes != "none":
        if notes == "full":
            bullets = [
                "Multi‑view fusion uses structural edges as a baseline; other views contribute gated deviations.",
                "V_abs is strict leave‑one‑out: computed from co‑portfolio neighbors excluding the anchor.",
                "Residual subtraction removes portfolio drift: z = z_pre − σ(pf_gate)·proj(V_abs).",
                "Market/Trade encoders add exogenous context; within‑portfolio attention models interactions inside a portfolio group.",
                "Optional portfolio head exposes V_abs directly to the head; optional negative drag penalizes alignment with V_abs.",
            ]
        else:  # short
            bullets = [
                "Fusion: structural baseline + gated per‑view deviations.",
                "Strict‑LOO V_abs excludes the anchor.",
                "Predict from residual anchor (portfolio de‑biased).",
                "Optional: market/trade context, within‑portfolio attention, negative drag.",
            ]

        li = "<BR ALIGN='LEFT'/>".join([f"• {b}" for b in bullets])
        notes_label = (
            "<"
            "<TABLE BORDER='0' CELLBORDER='0' CELLPADDING='2'>"
            "<TR><TD ALIGN='LEFT'><B>Legend</B></TD></TR>"
            f"<TR><TD ALIGN='LEFT'>{li}</TD></TR>"
            "</TABLE>"
            ">"
        )

        with g.subgraph(name="cluster_notes") as c:
            c.attr(label="", color="#94A3B8", style="rounded")
            c.node(
                "Notes",
                notes_label,
                shape="plain",
            )

        # Sink notes at bottom for TB; for LR it will typically sit to the right.
        with g.subgraph(name="rank_notes") as rn:
            rn.attr(rank="sink")
            rn.node("Notes")

        # Use invisible edges to bias notes placement without adding clutter
        g.edge("Head", "Notes", style="invis")

    # Emit DOT alongside the rendered artifact (portable)
    dot_path = Path(out_stem + ".dot")
    try:
        g.save(filename=str(dot_path))
    except Exception:
        pass

    # Render (or fall back to DOT)
    try:
        out_path = Path(g.render(out_stem, format=format, cleanup=True))
        return out_path
    except Exception as e:
        typer.echo(
            f"[warn] Render failed ({e}). Saved DOT instead: {dot_path}. "
            "Install Graphviz system binary to render: https://graphviz.org/download/"
        )
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
        "ppt16x9",
        help="Layout preset: ppt4x3 | ppt16x9 | square. Explicit width/height/dpi override the preset.",
    ),
    format: Optional[str] = typer.Option(
        None,
        help="Diagram format: svg | png | pdf. If not set, derived from output extension or defaults to svg.",
    ),
    force_market: Optional[bool] = typer.Option(
        None,
        help="Override: include Market Encoder (default inferred from model_config).",
    ),
    force_trade: Optional[bool] = typer.Option(
        None,
        help="Override: include Trade Encoder (default inferred from model_config).",
    ),
    force_portfolio_attn: Optional[bool] = typer.Option(
        None,
        help="Override: include Within‑Portfolio Attention block (default inferred from model_config).",
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
        None, help="Graph layout direction: TB (top→bottom) or LR (left→right). Preset defaults to TB."
    ),
    square: bool = typer.Option(
        False, help="Force a square canvas if width/height are not provided."
    ),
    width: Optional[float] = typer.Option(
        None, help="Target diagram width in inches (Graphviz 'size')."
    ),
    height: Optional[float] = typer.Option(
        None, help="Target diagram height in inches (Graphviz 'size')."
    ),
    dpi: Optional[int] = typer.Option(
        None, help="Raster DPI for PNG/JPG export (Graphviz 'dpi' graph attribute)."
    ),
    compact: bool = typer.Option(
        True, help="Use tighter spacing to better fit slides."
    ),
    notes: str = typer.Option(
        "short",
        help="Legend panel: none | short | full.",
    ),
):
    """Produce a clean, presentation‑quality MV‑DGT architecture diagram (Graphviz)."""
    workdir = Path(workdir)
    if not workdir.exists():
        raise typer.BadParameter(f"workdir does not exist: {workdir}")

    # Validate presence; contents aren't required for the conceptual diagram
    _ = _load_meta(workdir)
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

    _require_graphviz()

    # Apply preset defaults
    width2, height2, dpi2, rankdir2, square2, notes2 = _apply_preset(
        preset=preset,
        fmt=fmt,
        width=width,
        height=height,
        dpi=dpi,
        rankdir=rankdir,
        square=square,
        notes=notes,
    )

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
        rankdir=rankdir2 or "TB",
        square=square2,
        width=width2,
        height=height2,
        dpi=dpi2,
        compact=compact,
        notes=notes2 or "short",
    )

    typer.echo(f"[ok] MV‑DGT architecture diagram saved to: {path}")


if __name__ == "__main__":
    app()