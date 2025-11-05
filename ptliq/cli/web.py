from __future__ import annotations
import os
import signal
from pathlib import Path
import atexit
import typer
from rich import print
import logging
from ptliq.web.site import build_ui

PID_ENV = "PTLIQ_WEB_PIDFILE"

app = typer.Typer(no_args_is_help=False)


def _pidfile_path(port: int) -> Path:
    # Allow tests or power users to override via env var
    override = os.environ.get(PID_ENV)
    if override:
        return Path(override)
    home = Path.home() / ".ptliq"
    home.mkdir(parents=True, exist_ok=True)
    return home / f"ptliq-web-{port}.pid"


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but different user/session; treat as alive
        return True


@app.callback(invoke_without_command=True)
def app_main(
    ctx: typer.Context,
    api_url: str = typer.Option("http://127.0.0.1:8011", help="Base URL of the FastAPI scoring service"),
    host: str = typer.Option("127.0.0.1", help="Host/interface to bind the UI"),
    port: int = typer.Option(7861, help="Port for the UI server"),
    open_browser: bool = typer.Option(True, help="Open the UI in browser (use --no-open-browser to disable)"),
    share: bool = typer.Option(False, help="Create a public share URL (requires external network)"),
    force: bool = typer.Option(False, help="Start even if an existing pidfile is present (may replace a stale one)"),
    model_dir: Path = typer.Option(Path("models/dgt0"), help="Path to model run directory; if it contains an 'out' subdir with samples.parquet, that will be used."),
    verbose: bool = typer.Option(False, help="Enable verbose logging (diagnostics about samples lookup)"),
):
    """
    Launch a minimal website (Gradio) for submitting JSON payloads to the scoring API
    and viewing the results in a filterable grid.
    If a subcommand is invoked (e.g., `stop`), this callback will not launch the UI.
    """
    # If subcommand is being invoked, skip starting the server
    if ctx.invoked_subcommand is not None:
        return

    pidfile = _pidfile_path(port)
    if pidfile.exists() and not force:
        try:
            pid = int(pidfile.read_text().strip())
        except Exception:
            pid = None
        if pid and _process_alive(pid):
            print(f"[yellow]ptliq-web already running[/yellow] (pid={pid}, pidfile={pidfile}). Use --force to overwrite or run 'ptliq-web stop --port {port}'.")
            raise typer.Exit(code=1)
        else:
            print(f"[yellow]Removing stale pidfile[/yellow]: {pidfile}")
            try:
                pidfile.unlink(missing_ok=True)
            except Exception:
                pass

    # Write pidfile for stop command
    try:
        pidfile.write_text(str(os.getpid()))
    except Exception:
        # Non-fatal: proceed without pidfile
        pass

    def _cleanup():
        try:
            if pidfile.exists():
                pidfile.unlink()
        except Exception:
            pass

    atexit.register(_cleanup)

    # Harden Gradio 4.15.0 against environments without external network
    # Use local assets and disable analytics to avoid long waits/spinners.
    os.environ.setdefault("GRADIO_USE_CDN", "false")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("ptliq.web.cli")

    # Resolve a workdir that contains samples.parquet (prefer <model_dir>/out)
    resolved_workdir = None
    try:
        md = Path(model_dir)
        logger.info(f"model_dir provided: {md}")
        cand1 = md / "out"
        cand2 = md
        cand1_exists = (cand1 / "samples.parquet").exists()
        cand2_exists = (cand2 / "samples.parquet").exists()
        logger.info(f"checking samples: {cand1/'samples.parquet'} exists={cand1_exists}")
        logger.info(f"checking samples: {cand2/'samples.parquet'} exists={cand2_exists}")
        if cand1_exists:
            resolved_workdir = cand1
            logger.info(f"resolved workdir: {resolved_workdir}")
        elif cand2_exists:
            resolved_workdir = cand2
            logger.info(f"resolved workdir: {resolved_workdir}")
        else:
            logger.warning("samples.parquet not found under model_dir; UI will use fallback example payload")
    except Exception as e:
        logger.exception(f"error while resolving model_dir: {e}")
        resolved_workdir = None

    if resolved_workdir is not None:
        ui = build_ui(api_url, workdir=resolved_workdir)
    else:
        ui = build_ui(api_url)
    # Avoid any external calls or analytics that may hang in restricted networks.
    ui.launch(
        server_name=host,
        server_port=port,
        inbrowser=open_browser,
        share=share,
        show_api=False,
        prevent_thread_lock=False,
    )


@app.command("stop")
def stop_main(
    port: int = typer.Option(7861, help="Port of the UI server whose pidfile should be used"),
    timeout: float = typer.Option(5.0, help="Seconds to wait for graceful shutdown"),
):
    """
    Stop a running ptliq-web process by reading its pidfile and sending SIGTERM.
    """
    pidfile = _pidfile_path(port)
    if not pidfile.exists():
        print(f"[yellow]No pidfile found[/yellow] at {pidfile}. Nothing to stop.")
        raise typer.Exit(code=0)

    try:
        pid = int(pidfile.read_text().strip())
    except Exception:
        print(f"[red]Invalid pidfile[/red] at {pidfile}. Removing.")
        pidfile.unlink(missing_ok=True)
        raise typer.Exit(code=1)

    if not _process_alive(pid):
        print(f"[yellow]Process {pid} is not running[/yellow]. Removing pidfile {pidfile}.")
        pidfile.unlink(missing_ok=True)
        raise typer.Exit(code=0)

    try:
        os.kill(pid, signal.SIGTERM)
    except PermissionError:
        print(f"[red]Permission denied[/red] to signal pid {pid}.")
        raise typer.Exit(code=1)
    except ProcessLookupError:
        print(f"[yellow]Process {pid} already exited[/yellow].")

    # Wait for termination
    import time
    start = time.time()
    while time.time() - start < timeout:
        if not _process_alive(pid):
            break
        time.sleep(0.1)

    if _process_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"[yellow]Sent SIGKILL to pid {pid} after timeout[/yellow].")
        except Exception:
            pass

    try:
        pidfile.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"[green]Stopped[/green] ptliq-web (pid={pid}).")


app = app

if __name__ == "__main__":
    app()
