
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import asdict
import json

import typer
from rich import print

from ptliq.utils.config import load_config, get_sim_config
from ptliq.utils.paths import ensure_dirs
from ptliq.utils.logging import setup_logging
from ptliq.data.simulate import SimParams, simulate
from ptliq.data.io import write_parquet

app = typer.Typer(no_args_is_help=True)


def _getattr_safe(obj, name: str, default):
    # supports both dot-access objects and dict-like
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    try:
        return obj[name]  # type: ignore[index]
    except Exception:
        return default


@app.command()
def main(
    config: Path = typer.Option(Path("configs/simulate.yaml"), help="YAML config with data.sim"),
    outdir: Path = typer.Option(Path("data/raw/sim"), help="Output directory"),
    seed: Optional[int] = typer.Option(None, help="Override seed"),

    # Existing knobs (with safe fallbacks)
    par: Optional[float] = typer.Option(None, help="Face value used to anchor clean_price (default 100.0)"),
    base_spread_bps: Optional[float] = typer.Option(None, help="Baseline clean-price spread in bps"),
    sector_spread_bps: Optional[float] = typer.Option(None, help="Sector tilt for clean price in bps per code step"),
    rating_spread_bps: Optional[float] = typer.Option(None, help="Rating tilt for clean price in bps per code step"),
    clean_price_noise_bps: Optional[float] = typer.Option(None, help="Idiosyncratic noise in clean_price (bps)"),

    liq_size_coeff: Optional[float] = typer.Option(None, help="Coefficient of size_z in y_bps"),
    liq_side_coeff: Optional[float] = typer.Option(None, help="Coefficient of side_sign in y_bps"),
    liq_sector_coeff: Optional[float] = typer.Option(None, help="Coefficient of sector code in y_bps"),
    liq_rating_coeff: Optional[float] = typer.Option(None, help="Coefficient of rating code in y_bps"),
    liq_eps_bps: Optional[float] = typer.Option(None, help="Residual noise in y_bps (bps)"),
    micro_price_noise_bps: Optional[float] = typer.Option(None, help="Microstructure price noise added to y_bps (bps)"),

    # New portfolio/TRACE/provider knobs
    portfolio_trade_share: Optional[float] = typer.Option(None, help="Share of trade lines in portfolios"),
    basket_size_min: Optional[int] = typer.Option(None, help="Minimum size of a portfolio basket (lines)"),
    basket_size_max: Optional[int] = typer.Option(None, help="Maximum size of a portfolio basket (lines)"),
    port_skew_mu: Optional[float] = typer.Option(None, help="Mean strength of portfolio delta (bps scale)"),
    port_skew_sigma: Optional[float] = typer.Option(None, help="Std of portfolio delta strength"),
    port_time_spread_sec: Optional[int] = typer.Option(None, help="Seconds spread around basket time"),

    asof_rate: Optional[float] = typer.Option(None, help="Probability of As/Of flag"),
    late_rate: Optional[float] = typer.Option(None, help="Probability of Late flag"),
    cancel_rate: Optional[float] = typer.Option(None, help="Probability of Cancel flag"),
    ats_rate: Optional[float] = typer.Option(None, help="Probability of ATS flag"),

    cap_ig: Optional[float] = typer.Option(None, help="TRACE cap for IG"),
    cap_hy: Optional[float] = typer.Option(None, help="TRACE cap for HY"),

    gp_intercept_bps: Optional[float] = typer.Option(None, help="Provider map intercept (bps)"),
    gp_slope_bps: Optional[float] = typer.Option(None, help="Provider map slope (bps per 100 score)"),

    base_intensity: Optional[float] = typer.Option(None, help="Baseline daily Poisson intensity per bond"),
    liq_to_intensity: Optional[float] = typer.Option(None, help="Intensity lift per unit of static vendor liq"),

    liq_urgency_coeff: Optional[float] = typer.Option(None, help="Coefficient of urgency in y_bps"),
    second_resolution: Optional[bool] = typer.Option(None, help="Use second-level timestamps like TRACE"),
    loglevel: str = typer.Option("INFO"),
):
    """
    Generate simulated bonds & trades with TRACE-like fields and portfolio-conditioned effects.

    Output:
      - bonds.parquet: bond statics + provider baselines (vendor_* and pi_ref_* columns)
      - trades.parquet: trades with y_bps decomposition, flags, and TRACE-ish fields
    """
    setup_logging(loglevel)
    cfg = load_config(config)
    ensure_dirs(cfg)

    sim_cfg = get_sim_config(cfg)  # object or dict with at least n_bonds, n_days, providers, seed

    # Required basics (present in your existing config)
    n_bonds = _getattr_safe(sim_cfg, "n_bonds", 300)
    n_days = _getattr_safe(sim_cfg, "n_days", 30)
    providers = _getattr_safe(sim_cfg, "providers", ["P1"])
    sim_seed = seed if seed is not None else _getattr_safe(sim_cfg, "seed", 42)

    # Construct defaults to read dataclass defaults
    from ptliq.data.simulate import SimParams as _Def
    _defaults = _Def(n_bonds=1, n_days=1, providers=["P"], seed=0, outdir=Path("."))

    def pick(name: str, cli_value, default_from=_defaults):
        return cli_value if cli_value is not None else _getattr_safe(sim_cfg, name, getattr(default_from, name))

    params = SimParams(
        n_bonds=int(n_bonds),
        n_days=int(n_days),
        providers=list(providers),
        seed=int(sim_seed),
        outdir=Path(outdir),

        par=float(pick("par", par)),
        base_spread_bps=float(pick("base_spread_bps", base_spread_bps)),
        sector_spread_bps=float(pick("sector_spread_bps", sector_spread_bps)),
        rating_spread_bps=float(pick("rating_spread_bps", rating_spread_bps)),
        clean_price_noise_bps=float(pick("clean_price_noise_bps", clean_price_noise_bps)),

        liq_size_coeff=float(pick("liq_size_coeff", liq_size_coeff)),
        liq_side_coeff=float(pick("liq_side_coeff", liq_side_coeff)),
        liq_sector_coeff=float(pick("liq_sector_coeff", liq_sector_coeff)),
        liq_rating_coeff=float(pick("liq_rating_coeff", liq_rating_coeff)),
        liq_eps_bps=float(pick("liq_eps_bps", liq_eps_bps)),
        micro_price_noise_bps=float(pick("micro_price_noise_bps", micro_price_noise_bps)),

        portfolio_trade_share=float(pick("portfolio_trade_share", portfolio_trade_share)),
        basket_size_min=int(pick("basket_size_min", basket_size_min)),
        basket_size_max=int(pick("basket_size_max", basket_size_max)),
        port_skew_mu=float(pick("port_skew_mu", port_skew_mu)),
        port_skew_sigma=float(pick("port_skew_sigma", port_skew_sigma)),
        port_time_spread_sec=int(pick("port_time_spread_sec", port_time_spread_sec)),

        asof_rate=float(pick("asof_rate", asof_rate)),
        late_rate=float(pick("late_rate", late_rate)),
        cancel_rate=float(pick("cancel_rate", cancel_rate)),
        ats_rate=float(pick("ats_rate", ats_rate)),

        cap_ig=float(pick("cap_ig", cap_ig)),
        cap_hy=float(pick("cap_hy", cap_hy)),

        gp_intercept_bps=float(pick("gp_intercept_bps", gp_intercept_bps)),
        gp_slope_bps=float(pick("gp_slope_bps", gp_slope_bps)),

        base_intensity=float(pick("base_intensity", base_intensity)),
        liq_to_intensity=float(pick("liq_to_intensity", liq_to_intensity)),

        liq_urgency_coeff=float(pick("liq_urgency_coeff", liq_urgency_coeff)),
        second_resolution=bool(pick("second_resolution", second_resolution)),
    )

    Path(outdir).mkdir(parents=True, exist_ok=True)
    logging.info("Using output dir: %s", outdir)

    frames = simulate(params)

    # Write reproducibility sidecar metadata
    try:
        meta = asdict(params)
        # make non-serializable types JSON-friendly
        if isinstance(meta.get("outdir"), Path):
            meta["outdir"] = str(meta["outdir"])  # type: ignore[index]
        meta["sim_version"] = "0.1.0"
        from datetime import datetime as _dt
        meta["generated_at_utc"] = _dt.utcnow().isoformat(timespec="seconds")
        with open(Path(outdir) / "_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        logging.warning("Could not write _meta.json sidecar metadata", exc_info=False)

    # Drop compatibility alias to avoid duplicate numeric columns on disk
    trades_out = frames["trades"].drop(columns=["price_clean_exec"], errors="ignore")

    paths = {
        "bonds": write_parquet(frames["bonds"], Path(outdir) / "bonds.parquet"),
        "trades": write_parquet(trades_out, Path(outdir) / "trades.parquet"),
    }
    print(f"[bold green]Done.[/bold green] Wrote:")
    for k, p in paths.items():
        print(f"  • {k}: {p}")


if __name__ == "__main__":
    app()