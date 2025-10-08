import logging
from pathlib import Path
import typer
from rich import print

from ptliq.utils.config import load_config, get_sim_config
from ptliq.utils.paths import ensure_dirs
from ptliq.utils.logging import setup_logging
from ptliq.data.simulate import SimParams, simulate
from ptliq.data.io import write_parquet

app = typer.Typer(no_args_is_help=True)

@app.command()
def main(
    config: Path = typer.Option(Path("configs/base.yaml"), help="YAML config with data.sim"),
    outdir: Path = typer.Option(Path("data/raw/sim"), help="Output directory"),
    seed: int | None = typer.Option(None, help="Override seed"),
    loglevel: str = typer.Option("INFO")
):
    setup_logging(loglevel)
    cfg = load_config(config)
    ensure_dirs(cfg)

    sim_cfg = get_sim_config(cfg)
    if seed is not None:
        sim_cfg.seed = seed

    run_dir = Path(outdir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Using output dir: %s", run_dir)
    frames = simulate(SimParams(
        n_bonds=sim_cfg.n_bonds,
        n_days=sim_cfg.n_days,
        providers=sim_cfg.providers,
        seed=sim_cfg.seed,
        outdir=run_dir
    ))

    paths = {
        "bonds": write_parquet(frames["bonds"], run_dir / "bonds.parquet"),
        "trades": write_parquet(frames["trades"], run_dir / "trades.parquet"),
    }
    print(f"[bold green]Done.[/bold green] Wrote:")
    for k, p in paths.items():
        print(f"  • {k}: {p}")

if __name__ == "__main__":
    app()
