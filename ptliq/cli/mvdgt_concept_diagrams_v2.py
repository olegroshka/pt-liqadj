from __future__ import annotations

"""Presentation-grade conceptual diagrams for MV-DGT.

v2 tweaks:
- Legends use a 2-column bullet table so bullets align perfectly.
- DGT fusion legend is moved *below* the diagram using a graph-level label.
- Fusion formula is wrapped to avoid text collisions.
- Slightly more spacing for cleaner orthogonal routing.
"""

from pathlib import Path
from typing import Iterable, Literal

import typer
from graphviz import Digraph


app = typer.Typer(no_args_is_help=True)

Preset = Literal["ppt16x9", "ppt4x3", "a4"]


# -----------------------------
# Style helpers
# -----------------------------
def _html_box(title: str, body: str, dims: str = "") -> str:
    """Compact HTML-like label with consistent typography."""
    dims_row = (
        f"<TR><TD ALIGN='LEFT'><FONT POINT-SIZE='10' COLOR='#64748B'>{dims}</FONT></TD></TR>"
        if dims
        else ""
    )
    return (
        "<"
        "<TABLE BORDER='0' CELLBORDER='0' CELLPADDING='2'>"
        f"<TR><TD ALIGN='LEFT'><B>{title}</B></TD></TR>"
        f"<TR><TD ALIGN='LEFT'>{body}</TD></TR>"
        f"{dims_row}"
        "</TABLE>"
        ">"
    )


def _apply_preset(g: Digraph, preset: Preset, rankdir: str = "TB") -> None:
    # Graph-level styling tuned for slides.
    base = dict(
        rankdir=rankdir,
        bgcolor="white",
        fontname="Inter,Helvetica,Arial",
        fontsize="16" if preset != "a4" else "14",
        pad="0.08" if preset != "ppt4x3" else "0.06",
        nodesep="0.38" if preset != "ppt4x3" else "0.35",
        ranksep="0.50" if preset != "ppt4x3" else "0.46",
        splines="ortho",
        concentrate="true",
    )
    g.graph_attr.update(base)
    g.node_attr.update(dict(fontname="Inter,Helvetica,Arial"))
    g.edge_attr.update(dict(color="#334155", arrowsize="0.8"))


def _legend_items(mode: Literal["none", "short", "full"]) -> list[str]:
    if mode == "none":
        return []
    if mode == "short":
        return [
            "D = hidden size (shared embedding)",
            "LOO = strict leave‑one‑out portfolio prototype",
            "Gates use σ(·) = sigmoid",
        ]
    return [
        "Graph views: struct, port, corr_global, corr_local",
        "Differential fusion treats struct as baseline; other views add gated deviations",
        "Strict LOO excludes the anchor itself; V_abs is L2‑normalized in H‑space",
        "Residual fusion: z = z_pre − σ(pf_gate) · pf_proj([V_abs, 0])",
        "Negative drag (optional): subtracts |cos(z_pre, V_abs)| scaled by a coefficient",
    ]


def _legend_html_table(items: Iterable[str], title: str = "Legend") -> str:
    rows = "".join(
        f"<TR><TD ALIGN='LEFT' WIDTH='14'>•</TD><TD ALIGN='LEFT'>{it}</TD></TR>" for it in items
    )
    inner = (
        "<TABLE BORDER='0' CELLBORDER='0' CELLPADDING='0' CELLSPACING='2'>"
        f"{rows}"
        "</TABLE>"
    )
    return (
        "<"
        "<TABLE BORDER='0' CELLBORDER='1' CELLPADDING='6' BGCOLOR='#F8FAFC' COLOR='#CBD5E1'>"
        f"<TR><TD ALIGN='LEFT' BGCOLOR='#F1F5F9'><B>{title}</B></TD></TR>"
        f"<TR><TD ALIGN='LEFT'>{inner}</TD></TR>"
        "</TABLE>"
        ">"
    )


def _legend_node(g: Digraph, node_id: str, mode: Literal["none", "short", "full"]) -> None:
    items = _legend_items(mode)
    if not items:
        return
    g.node(node_id, _legend_html_table(items), shape="plain")


# -----------------------------
# Diagram 1: "Signal extraction"
# -----------------------------
def diagram_signal_extraction(
    preset: Preset,
    legend: Literal["none", "short", "full"],
    show_optional_blocks: bool = True,
    fmt: str = "svg",
) -> Digraph:
    g = Digraph("MV_DGT_SignalExtraction", format=fmt)
    _apply_preset(g, preset, rankdir="TB")

    g.node(
        "TITLE",
        _html_box(
            "MV‑DGT — signal extraction pipeline",
            "Multi‑view graph → anchor residual → optional context/attention → regression",
        ),
        shape="plain",
    )

    with g.subgraph(name="cluster_inputs") as c:
        c.attr(label="Inputs", color="#94A3B8", style="rounded", penwidth="1.2")
        c.node("X", _html_box("Node features", "x", "[N, x_dim]"), shape="plain")
        c.node("EI", _html_box("Graph buffers", "edge_index / edge_weight", "[2,E], [E]"), shape="plain")
        c.node("MASKS", _html_box("View masks", "struct / port / corrG / corrL", "per‑view edge subsets"), shape="plain")
        c.node("ANCH", _html_box("Batch anchors", "anchor_idx", "[B]"), shape="plain")
        c.node("PFGID", _html_box("Portfolio id", "pf_gid", "[B] (−1 means none)"), shape="plain")
        c.node("PFC", _html_box("Portfolio context", "port_ctx (flat)", "nodes + weights + lengths"), shape="plain")

    with g.subgraph(name="cluster_views") as c:
        c.attr(label="Graph views (how edges are defined)", color="#64748B", style="rounded", penwidth="1.2")
        c.node("V_STRUCT", _html_box("Structural view", "static edges", "mask_struct"), shape="plain")
        c.node("V_PORT", _html_box("Portfolio view", "co‑membership edges", "mask_port"), shape="plain")
        c.node("V_CG", _html_box("Corr view (global)", "correlation edges", "mask_corr_global"), shape="plain")
        c.node("V_CL", _html_box("Corr view (local)", "correlation edges", "mask_corr_local"), shape="plain")
        c.node("STD", _html_box("Per‑view standardize", "weights → comparable scale", "non‑struct views"), shape="plain")
        c.node("CORRG", _html_box("corr_gate", "soft down‑weight corr views", "σ(init≈−1)"), shape="plain")
        c.edges([("V_PORT", "STD"), ("V_CG", "STD"), ("V_CL", "STD")])
        c.edge("STD", "CORRG")

    with g.subgraph(name="cluster_graph") as c:
        c.attr(label="Multi‑view Graph Encoder (2 layers)", color="#4C78A8", style="rounded", penwidth="1.3")
        c.node("ENC", _html_box("Node encoder", "MLP + LayerNorm", "x_dim → D"), shape="plain")
        c.node("L1", _html_box("DGT layer 1", "TransformerConv per view + fusion", "learn σ(g1[v])"), shape="plain")
        c.node("L2", _html_box("DGT layer 2", "TransformerConv per view + fusion", "learn σ(g2[v])"), shape="plain")
        c.node("H", _html_box("Node embeddings", "H", "[N, D]"), shape="plain")
        c.edges([("ENC", "L1"), ("L1", "L2"), ("L2", "H")])

    with g.subgraph(name="cluster_anchor") as c:
        c.attr(label="Anchor residual (portfolio‑conditioned)", color="#F58518", style="rounded", penwidth="1.3")
        c.node("GATHER", _html_box("Gather anchor", "z_pre = H[anchor_idx]", "[B, D]"), shape="plain")
        c.node("LOO", _html_box("Strict LOO prototype", "V_abs from co‑portfolio neighbors", "L2‑norm, [B, D]"), shape="plain")
        c.node("PFP", _html_box("Residual fusion", "z = z_pre − σ(pf_gate)·pf_proj([V_abs,0])", "2D → D"), shape="plain")
        c.edges([("H", "GATHER"), ("H", "LOO"), ("GATHER", "PFP"), ("LOO", "PFP")])

    with g.subgraph(name="cluster_ctx") as c:
        c.attr(label="Context & within‑portfolio interactions (optional)", color="#54A24B", style="rounded", penwidth="1.3")
        c.node("MKT", _html_box("Market encoder", "mkt_enc", "market_feat → D"), shape="plain")
        c.node("TRD", _html_box("Trade encoder", "trade_enc", "trade_feat → D"), shape="plain")
        c.node("ATTN", _html_box("Portfolio self‑attention", "tokens grouped by pf_gid", "TransformerEncoder"), shape="plain")
        if not show_optional_blocks:
            c.node("CTX_COLLAPSE", _html_box("Optional branches", "market / trade / portfolio‑attn", ""), shape="plain")

    with g.subgraph(name="cluster_head") as c:
        c.attr(label="Prediction", color="#0F172A", style="rounded", penwidth="1.3")
        c.node("CAT", _html_box("Concatenate", "[z, (ctx...), (pf_head...), (attn_ctx...)]", ""), shape="plain")
        c.node("HEAD", _html_box("Regression head", "MLP → ŷ", "[B]"), shape="plain")
        c.edge("CAT", "HEAD")

    g.node("DRAG", _html_box("Negative drag (opt)", "− λ · |cos(z_pre, V_abs)|", "conservative bias"), shape="plain")

    # Wiring across clusters
    g.edge("TITLE", "X", style="invis")
    g.edge("X", "ENC")
    g.edge("EI", "V_STRUCT")
    g.edge("MASKS", "V_STRUCT")
    g.edges([("V_STRUCT", "L1"), ("V_PORT", "L1"), ("V_CG", "L1"), ("V_CL", "L1")])
    g.edge("ANCH", "GATHER")
    g.edge("PFGID", "LOO")
    g.edge("PFC", "LOO")

    if show_optional_blocks:
        g.edge("PFP", "CAT")
        g.edge("MKT", "CAT")
        g.edge("TRD", "CAT")
        g.edge("PFP", "ATTN")
        g.edge("MKT", "ATTN")
        g.edge("TRD", "ATTN")
        g.edge("ATTN", "CAT")
    else:
        g.edge("PFP", "CTX_COLLAPSE")
        g.edge("CTX_COLLAPSE", "CAT")

    g.edge("GATHER", "DRAG", style="dashed")
    g.edge("LOO", "DRAG", style="dashed")
    g.edge("DRAG", "CAT", style="dashed")

    _legend_node(g, "LEGEND", legend)
    if _legend_items(legend):
        g.edge("HEAD", "LEGEND", style="invis")

    return g


# -----------------------------
# Diagram 2: DGT fusion layer
# -----------------------------
def diagram_dgt_fusion_layer(
    preset: Preset,
    legend: Literal["none", "short", "full"],
    fmt: str = "svg",
) -> Digraph:
    g = Digraph("MV_DGT_DGTFusion", format=fmt)
    _apply_preset(g, preset, rankdir="LR")
    # Give this diagram a bit more air; orthogonal arrows route cleaner.
    g.graph_attr.update(dict(nodesep="0.48", ranksep="0.65"))

    g.node("TITLE", _html_box("DGT layer (one graph layer)", "Differential fusion over 4 views", ""), shape="plain")
    g.node("XH", _html_box("Input features", "h_in", "[N, D]"), shape="plain")

    with g.subgraph(name="cluster_views") as c:
        c.attr(label="Per‑view TransformerConv", color="#4C78A8", style="rounded", penwidth="1.3")
        c.attr(rank="same")
        c.node("S", _html_box("struct", "h_s = Conv_struct(h_in)", "baseline"), shape="plain")
        c.node("P", _html_box("port", "h_p = Conv_port(h_in)", "co‑membership"), shape="plain")
        c.node("CG", _html_box("corr_global", "h_cg = Conv_cg(h_in)", "corr_gate·std(w)"), shape="plain")
        c.node("CL", _html_box("corr_local", "h_cl = Conv_cl(h_in)", "corr_gate·std(w)"), shape="plain")

    g.edges([("XH", "S"), ("XH", "P"), ("XH", "CG"), ("XH", "CL")])

    with g.subgraph(name="cluster_delta") as c:
        c.attr(label="Gated deviations (vs struct)", color="#64748B", style="rounded", penwidth="1.2")
        c.node("DP", _html_box("Δ_port", "h_p − h_s", ""), shape="plain")
        c.node("DCG", _html_box("Δ_corrG", "h_cg − h_s", ""), shape="plain")
        c.node("DCL", _html_box("Δ_corrL", "h_cl − h_s", ""), shape="plain")
        c.node("GP", _html_box("gate_port", "σ(g[port])", "scalar per layer"), shape="plain")
        c.node("GCG", _html_box("gate_corrG", "σ(g[corrG])", "scalar per layer"), shape="plain")
        c.node("GCL", _html_box("gate_corrL", "σ(g[corrL])", "scalar per layer"), shape="plain")
        c.node("GS", _html_box("gate_struct", "σ(g[struct])", "scalar per layer"), shape="plain")

    g.edge("P", "DP")
    g.edge("S", "DP")
    g.edge("CG", "DCG")
    g.edge("S", "DCG")
    g.edge("CL", "DCL")
    g.edge("S", "DCL")

    # Wrapped fusion expression (prevents long single-line boxes)
    fuse_body = (
        "h_out = h_in"  # line 1
        "<BR/>+ GS·h_s"  # line 2
        "<BR/>+ GP·(h_p − h_s)"  # line 3
        "<BR/>+ GCG·(h_cg − h_s)"  # line 4
        "<BR/>+ GCL·(h_cl − h_s)"  # line 5
    )
    g.node("FUSE", _html_box("Fuse + residual", fuse_body, ""), shape="plain")

    g.edges([("S", "FUSE"), ("DP", "FUSE"), ("DCG", "FUSE"), ("DCL", "FUSE")])
    g.edges([("GS", "FUSE"), ("GP", "FUSE"), ("GCG", "FUSE"), ("GCL", "FUSE")])
    # With orthogonal routing, edge labels can float oddly; xlabel tends to behave better.
    g.edge("XH", "FUSE", xlabel="residual", fontsize="10", fontcolor="#64748B", color="#64748B")

    g.node("NORM", _html_box("Norm", "LayerNorm / dropout (impl‑dependent)", ""), shape="plain")
    g.edge("FUSE", "NORM")

    # Legend below: use graph label anchored at bottom.
    items = _legend_items(legend)
    if items:
        g.graph_attr.update(
            dict(
                labelloc="b",
                labeljust="l",
                label=_legend_html_table(items, title="Legend"),
            )
        )

    return g


# -----------------------------
# Diagram 3: strict LOO prototype
# -----------------------------
def diagram_strict_loo(
    preset: Preset,
    legend: Literal["none", "short", "full"],
    fmt: str = "svg",
) -> Digraph:
    g = Digraph("MV_DGT_StrictLOO", format=fmt)
    _apply_preset(g, preset, rankdir="TB")

    g.node("TITLE", _html_box("Strict LOO portfolio prototype", "How MV‑DGT removes co‑portfolio drift", ""), shape="plain")

    with g.subgraph(name="cluster_in") as c:
        c.attr(label="Given", color="#94A3B8", style="rounded", penwidth="1.2")
        c.node("ZPRE", _html_box("Anchor embedding", "z_pre = H[anchor_idx]", "[B, D]"), shape="plain")
        c.node("PORT", _html_box("Co‑portfolio items", "node ids + signed weights", "from port_ctx + pf_gid"), shape="plain")
        c.node("H", _html_box("Node embeddings", "H", "[N, D]"), shape="plain")

    with g.subgraph(name="cluster_loo") as c:
        c.attr(label="LOO prototype in H‑space", color="#F58518", style="rounded", penwidth="1.3")
        c.node("EXCL", _html_box("Exclude anchor", "strict LOO (never include anchor)", ""), shape="plain")
        c.node("SUM", _html_box("Weighted sum", "Σ |w_i| · H[i]", "absolute weights"), shape="plain")
        c.node("NORM", _html_box("Normalize", "V_abs = L2‑norm(sum)", "[B, D]"), shape="plain")
        c.edges([("EXCL", "SUM"), ("SUM", "NORM")])

    with g.subgraph(name="cluster_res") as c:
        c.attr(label="Residual fusion (forward path)", color="#4C78A8", style="rounded", penwidth="1.3")
        c.node("PROJ", _html_box("pf_proj", "[V_abs, 0] → D", "2D → D"), shape="plain")
        c.node("GATE", _html_box("pf_gate", "σ(pf_gate)", "scalar"), shape="plain")
        c.node("SUB", _html_box("Subtract", "z = z_pre − gate·proj", "[B, D]"), shape="plain")
        c.edges([("PROJ", "GATE"), ("GATE", "SUB")])

    with g.subgraph(name="cluster_drag") as c:
        c.attr(label="Optional conservative bias", color="#64748B", style="rounded", penwidth="1.2")
        c.node("COS", _html_box("Alignment", "|cos( normalize(z_pre), V_abs )|", ""), shape="plain")
        c.node("DRAG", _html_box("Negative drag", "ŷ ← ŷ − λ·alignment", "signless"), shape="plain")
        c.edge("COS", "DRAG", style="dashed")

    g.edge("H", "EXCL")
    g.edge("PORT", "EXCL")
    g.edge("NORM", "PROJ")
    g.edge("ZPRE", "SUB")
    g.edge("PROJ", "SUB")
    g.edge("ZPRE", "COS", style="dashed")
    g.edge("NORM", "COS", style="dashed")

    g.node("OUT", _html_box("Result", "Residual anchor z used by head", ""), shape="plain")
    g.edge("SUB", "OUT")

    _legend_node(g, "LEGEND", legend)
    if _legend_items(legend):
        g.edge("OUT", "LEGEND", style="invis")

    return g


def _render(g: Digraph, out: Path) -> Path:
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    return Path(g.render(str(stem), cleanup=True))


@app.command()
def build(
    outdir: Path = typer.Option(Path("./diagrams"), help="Output directory."),
    preset: Preset = typer.Option("ppt16x9", help="Layout tuning target."),
    format: str = typer.Option("svg", help="svg (best for slides) or png."),
    legend: Literal["none", "short", "full"] = typer.Option("short", help="Legend verbosity."),
    optional: bool = typer.Option(True, help="Show optional blocks explicitly (market/trade/attn)."),
):
    """Build three conceptual diagrams."""

    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    g1 = diagram_signal_extraction(preset=preset, legend=legend, show_optional_blocks=optional, fmt=format)
    g2 = diagram_dgt_fusion_layer(preset=preset, legend=legend, fmt=format)
    g3 = diagram_strict_loo(preset=preset, legend=legend, fmt=format)

    p1 = _render(g1, outdir / f"mv_dgt_signal_extraction.{format}")
    p2 = _render(g2, outdir / f"mv_dgt_dgt_fusion_layer.{format}")
    p3 = _render(g3, outdir / f"mv_dgt_strict_loo.{format}")

    typer.echo(f"Wrote:\n- {p1}\n- {p2}\n- {p3}")


if __name__ == "__main__":
    app()
