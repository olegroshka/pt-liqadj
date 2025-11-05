from __future__ import annotations
from pathlib import Path
import os
import signal
import atexit
import typer
import uvicorn
from rich import print
from ptliq.service.scoring import MLPScorer, DGTScorer
from ptliq.service.app import create_app

PID_ENV = "PTLIQ_SERVE_PIDFILE"

app = typer.Typer(no_args_is_help=False)


def _pidfile_path(port: int) -> Path:
    override = os.environ.get(PID_ENV)
    if override:
        return Path(override)
    home = Path.home() / ".ptliq"
    home.mkdir(parents=True, exist_ok=True)
    return home / f"ptliq-serve-{port}.pid"


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@app.callback(invoke_without_command=True)
def app_main(
    ctx: typer.Context,
    package: Path = typer.Option(Path("serving/tmp_model"), help="Path to models/<run_id> or serving/packages/<run_id>.zip"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8011),
    device: str = typer.Option("cpu"),
    model: str = typer.Option("mlp", help="Which model to serve: 'mlp' (packaged) or 'dgt' (graph workdir)"),
    force: bool = typer.Option(False, help="Start even if an existing pidfile is present"),
):
    """
    Serve the model with FastAPI. Accepts either a model directory or a packaged zip.
    Creates a pidfile so it can be stopped via `ptliq-serve stop`.
    If a subcommand is invoked (e.g., `stop`), this callback will not start the server.
    """
    # If a subcommand (like stop) is being called, do nothing here
    if ctx.invoked_subcommand is not None:
        return

    if package is None:
        raise typer.BadParameter("--package is required when starting the server")

    pidfile = _pidfile_path(port)
    if pidfile.exists() and not force:
        try:
            pid = int(pidfile.read_text().strip())
        except Exception:
            pid = None
        if pid and _process_alive(pid):
            print(f"[yellow]ptliq-serve already running[/yellow] (pid={pid}, pidfile={pidfile}). Use --force to overwrite or run 'ptliq-serve stop --port {port}'.")
            raise typer.Exit(code=1)
        else:
            print(f"[yellow]Removing stale pidfile[/yellow]: {pidfile}")
            try:
                pidfile.unlink(missing_ok=True)
            except Exception:
                pass

    try:
        pidfile.write_text(str(os.getpid()))
    except Exception:
        pass

    def _cleanup():
        try:
            if pidfile.exists():
                pidfile.unlink()
        except Exception:
            pass

    atexit.register(_cleanup)

    package = Path(package)
    m = (model or "").strip().lower()
    if m == "dgt":
        if not package.is_dir():
            print("[red]For model='dgt', --package must be a directory containing graph artifacts (workdir).")
            raise typer.Exit(code=2)
        scorer = DGTScorer.from_dir(package, device=device)
    else:
        # default to MLP packaged model
        if package.is_dir():
            scorer = MLPScorer.from_dir(package, device=device)
        else:
            scorer = MLPScorer.from_zip(package, device=device)
    api = create_app(scorer)
    uvicorn.run(api, host=host, port=port, log_level="info")


@app.command("stop")
def stop_main(
    port: int = typer.Option(8011, help="Port of the serve process whose pidfile should be used"),
    timeout: float = typer.Option(5.0, help="Seconds to wait for graceful shutdown"),
):
    """Stop a running ptliq-serve process by reading its pidfile and sending SIGTERM."""
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

    print(f"[green]Stopped[/green] ptliq-serve (pid={pid}).")

app = app

if __name__ == "__main__":
    app()
