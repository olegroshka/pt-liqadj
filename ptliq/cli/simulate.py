from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

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
    config: Path = typer.Option(Path("configs/base.yaml"), help="YAML config with data.sim"),
    outdir: Path = typer.Option(Path("data/raw/sim"), help="Output directory"),
    seed: Optional[int] = typer.Option(None, help="Override seed"),
    # Optional CLI overrides for new parameters (all keep existing defaults if omitted)
    par: float = typer.Option(None, help="Face value used to anchor clean_price (default 100.0)"),
    base_spread_bps: float = typer.Option(None, help="Baseline clean-price spread in bps"),
    sector_spread_bps: float = typer.Option(None, help="Sector tilt for clean price in bps per code step"),
    rating_spread_bps: float = typer.Option(None, help="Rating tilt for clean price in bps per code step"),
    clean_price_noise_bps: float = typer.Option(None, help="Idiosyncratic noise in clean_price (bps)"),
    liq_size_coeff: float = typer.Option(None, help="Coefficient of size_z in y_bps"),
    liq_side_coeff: float = typer.Option(None, help="Coefficient of side_sign in y_bps"),
    liq_sector_coeff: float = typer.Option(None, help="Coefficient of sector code in y_bps"),
    liq_rating_coeff: float = typer.Option(None, help="Coefficient of rating code in y_bps"),
    liq_eps_bps: float = typer.Option(None, help="Residual noise in y_bps (bps)"),
    micro_price_noise_bps: float = typer.Option(None, help="Microstructure price noise added to y_bps (bps)"),
    loglevel: str = typer.Option("INFO"),
):
    """
    Generate simulated bonds & trades:
      - bonds include sector/rating and clean_price
      - trades include y_bps (truth) and observed price constructed from clean_price and y_bps
    """
    setup_logging(loglevel)
    cfg = load_config(config)
    ensure_dirs(cfg)

    # `get_sim_config` keeps existing behavior; we read extras with safe fallbacks.
    sim_cfg = get_sim_config(cfg)  # object or dict with at least n_bonds, n_days, providers, seed

    # Required basics (present in your existing config)
    n_bonds = _getattr_safe(sim_cfg, "n_bonds", 300)
    n_days = _getattr_safe(sim_cfg, "n_days", 30)
    providers = _getattr_safe(sim_cfg, "providers", ["P1"])
    sim_seed = seed if seed is not None else _getattr_safe(sim_cfg, "seed", 42)

    # New knobs (fall back to SimParams defaults if not provided in config/file/CLI)
    sp_defaults = SimParams(
        n_bonds=1, n_days=1, providers=["P"], seed=0, outdir=Path(".")
    )  # just to access defaults

    def pick(name: str, cli_value, default_from=sp_defaults):
        return cli_value if cli_value is not None else _getattr_safe(sim_cfg, name, getattr(default_from, name))

    par_v = pick("par", par)
    base_spread_bps_v = pick("base_spread_bps", base_spread_bps)
    sector_spread_bps_v = pick("sector_spread_bps", sector_spread_bps)
    rating_spread_bps_v = pick("rating_spread_bps", rating_spread_bps)
    clean_price_noise_bps_v = pick("clean_price_noise_bps", clean_price_noise_bps)
    liq_size_coeff_v = pick("liq_size_coeff", liq_size_coeff)
    liq_side_coeff_v = pick("liq_side_coeff", liq_side_coeff)
    liq_sector_coeff_v = pick("liq_sector_coeff", liq_sector_coeff)
    liq_rating_coeff_v = pick("liq_rating_coeff", liq_rating_coeff)
    liq_eps_bps_v = pick("liq_eps_bps", liq_eps_bps)
    micro_price_noise_bps_v = pick("micro_price_noise_bps", micro_price_noise_bps)

    run_dir = Path(outdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Using output dir: %s", run_dir)

    params = SimParams(
        n_bonds=int(n_bonds),
        n_days=int(n_days),
        providers=list(providers),
        seed=int(sim_seed),
        outdir=run_dir,
        par=float(par_v),
        base_spread_bps=float(base_spread_bps_v),
        sector_spread_bps=float(sector_spread_bps_v),
        rating_spread_bps=float(rating_spread_bps_v),
        clean_price_noise_bps=float(clean_price_noise_bps_v),
        liq_size_coeff=float(liq_size_coeff_v),
        liq_side_coeff=float(liq_side_coeff_v),
        liq_sector_coeff=float(liq_sector_coeff_v),
        liq_rating_coeff=float(liq_rating_coeff_v),
        liq_eps_bps=float(liq_eps_bps_v),
        micro_price_noise_bps=float(micro_price_noise_bps_v),
    )

    frames = simulate(params)

    paths = {
        "bonds": write_parquet(frames["bonds"], run_dir / "bonds.parquet"),
        "trades": write_parquet(frames["trades"], run_dir / "trades.parquet"),
    }
    print(f"[bold green]Done.[/bold green] Wrote:")
    for k, p in paths.items():
        print(f"  • {k}: {p}")


if __name__ == "__main__":
    app()
