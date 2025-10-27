# ptliq/cli/tensorboard_stop.py
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(no_args_is_help=True)


def _which(exe: str) -> Optional[str]:
    from shutil import which
    return which(exe)


def _kill_pid(pid: int, force: bool = False) -> bool:
    try:
        if os.name == "nt":
            # Windows: use taskkill
            args = ["taskkill", "/PID", str(pid), "/F"] if force else ["taskkill", "/PID", str(pid)]
            return subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
        else:
            os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
            return True
    except Exception:
        return False


def _pids_from_port(port: int) -> List[int]:
    pids: List[int] = []
    if os.name != "nt":
        # Try lsof
        lsof = _which("lsof")
        if lsof:
            try:
                out = subprocess.check_output([lsof, "-t", f"-i:{port}"])
                for line in out.decode().strip().split():
                    if line.isdigit():
                        pids.append(int(line))
            except Exception:
                pass
        # Try fuser
        if not pids:
            fuser = _which("fuser")
            if fuser:
                try:
                    out = subprocess.check_output([fuser, "-n", "tcp", str(port)])
                    for tok in out.decode().strip().split():
                        tok = tok.strip()
                        if tok.isdigit():
                            pids.append(int(tok))
                except Exception:
                    pass
        # Fallback: pgrep with command line filter
        if not pids:
            pgrep = _which("pgrep")
            if pgrep:
                try:
                    pattern = f"tensorboard.*--port {port}"
                    out = subprocess.check_output([pgrep, "-f", pattern])
                    for line in out.decode().strip().split():
                        if line.isdigit():
                            pids.append(int(line))
                except Exception:
                    pass
    else:
        # Windows: best-effort using netstat (expensive); omit for now
        pass
    return sorted(set(pids))


@app.command()
def stop(
    pid: Optional[int] = typer.Option(None, help="PID to terminate (preferred)."),
    port: Optional[int] = typer.Option(None, help="Port TensorBoard is serving on (tries to discover PID and stop it)."),
    all: bool = typer.Option(False, help="Stop all tensorboard processes for current user (best effort)."),
    force: bool = typer.Option(False, help="Force kill if graceful stop fails."),
    wait: float = typer.Option(1.0, help="Seconds to wait after SIGTERM before force kill."),
):
    """
    Stop a running TensorBoard process.

    Examples:
      ptliq-stop-tensor-bord --port 6006
      ptliq-stop-tensor-bord --pid 12345
      ptliq-stop-tensor-bord --all
    """
    stopped: List[int] = []

    if pid is not None:
        ok = _kill_pid(pid, force=False)
        if not ok and force:
            time.sleep(max(0.0, wait))
            ok = _kill_pid(pid, force=True)
        if ok:
            typer.secho(f"Stopped TensorBoard [pid={pid}]", fg=typer.colors.GREEN)
            raise typer.Exit(code=0)
        else:
            typer.secho(f"Failed to stop PID {pid}.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    if port is not None:
        pids = _pids_from_port(port)
        if not pids:
            typer.secho(f"No process found on port {port}.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)
        for p in pids:
            if _kill_pid(p, force=False):
                stopped.append(p)
        if stopped and force:
            time.sleep(max(0.0, wait))
            # Try to kill remaining
            rem = [p for p in pids if p not in stopped]
            for p in rem:
                if _kill_pid(p, force=True):
                    stopped.append(p)
        if stopped:
            typer.secho(f"Stopped PIDs on port {port}: {stopped}", fg=typer.colors.GREEN)
            raise typer.Exit(code=0)
        else:
            typer.secho(f"Failed to stop any process on port {port}.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    if all:
        if os.name == "nt":
            # Best effort: kill any 'tensorboard' processes
            ret = subprocess.call(["taskkill", "/IM", "tensorboard.exe", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Also attempt to kill python -m tensorboard.main
            subprocess.call(["taskkill", "/F", "/FI", "WINDOWTITLE eq tensorboard"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ok = (ret == 0)
        else:
            # pkill - only for current user by default
            pkill = _which("pkill")
            if pkill:
                ok = subprocess.call([pkill, "-f", "tensorboard"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
            else:
                ok = False
        if ok:
            typer.secho("Stopped all tensorboard processes (best effort).", fg=typer.colors.GREEN)
            raise typer.Exit(code=0)
        else:
            typer.secho("Failed to stop tensorboard processes. Try --port or --pid.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    # If no selector provided, default to common port 6006
    default_port = 6006
    pids = _pids_from_port(default_port)
    if not pids:
        typer.secho("No selector given and nothing found on :6006.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)
    for p in pids:
        if _kill_pid(p, force=False):
            stopped.append(p)
    if not stopped and force:
        time.sleep(max(0.0, wait))
        for p in pids:
            if _kill_pid(p, force=True):
                stopped.append(p)
    if stopped:
        typer.secho(f"Stopped TensorBoard on :{default_port} [pids={stopped}]", fg=typer.colors.GREEN)
        raise typer.Exit(code=0)
    else:
        typer.secho(f"Failed to stop TensorBoard on :{default_port}.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


# Keep Typer app export for entrypoint
