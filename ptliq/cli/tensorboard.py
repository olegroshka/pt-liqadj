# ptliq/cli/tensorboard.py
from __future__ import annotations

from pathlib import Path
import sys
import time
import webbrowser
import subprocess
from typing import Optional

import typer

app = typer.Typer(no_args_is_help=True)


def _which(exe: str) -> Optional[str]:
    from shutil import which
    return which(exe)


@app.command()
def start_main(
    logdir: Path = typer.Option(Path("models"), help="TensorBoard log directory (root)."),
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(6006, help="Port to use."),
    open_browser: bool = typer.Option(True, help="Open default web browser to the dashboard."),
    reload_interval: int = typer.Option(5, help="Reload interval seconds."),
    foreground: bool = typer.Option(False, help="Run in foreground (block). Default: spawn background process."),
):
    """
    Start a local TensorBoard server pointed at --logdir and open the browser.
    Defaults to background mode; use --foreground to block.
    """
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    url = f"http://{host}:{port}/"

    # Prefer module invocation to avoid PATH issues
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir), "--host", host, "--port", str(port), "--reload_interval", str(reload_interval)]

    if foreground:
        typer.echo(f"[tensorboard] starting in foreground: {' '.join(cmd)}")
        typer.echo(f"Open: {url}")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        # Run and block
        return_code = subprocess.call(cmd)
        raise typer.Exit(code=return_code)
    else:
        # Spawn detached process
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            # Fallback to 'tensorboard' executable if module resolution fails
            tb_bin = _which("tensorboard")
            if not tb_bin:
                typer.secho("tensorboard is not installed. Please `pip install tensorboard`.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            proc = subprocess.Popen([tb_bin, "--logdir", str(logdir), "--host", host, "--port", str(port), "--reload_interval", str(reload_interval)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Give it a moment to bind
        time.sleep(0.8)
        typer.secho(f"TensorBoard started [pid={proc.pid}] at {url}", fg=typer.colors.GREEN)
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        # Print hint to stop
        typer.echo("Use your OS task manager or `pkill -f tensorboard` to stop the background process.")
        raise typer.Exit(code=0)


# Keep Typer app export for entrypoint
# The script entrypoint points directly to start_main
