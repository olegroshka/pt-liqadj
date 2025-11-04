from __future__ import annotations
import os
import sys
import time
import subprocess
from pathlib import Path
from typer.testing import CliRunner

from ptliq.cli.serve import app as serve_app

runner = CliRunner(mix_stderr=False)


def test_serve_stop_no_pidfile(tmp_path: Path):
    # Point the CLI to a temp pidfile that does not exist
    env = os.environ.copy()
    pidfile = tmp_path / "pidfile.pid"
    env["PTLIQ_SERVE_PIDFILE"] = str(pidfile)

    res = runner.invoke(serve_app, ["stop"], env=env)
    # Should be a clean no-op
    assert res.exit_code == 0, res.stdout
    assert not pidfile.exists()


def test_serve_stop_kills_process(tmp_path: Path):
    # Spawn a child Python process that sleeps; we'll stop it via CLI
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert sleeper.poll() is None  # running
        # write its pid to a temp pidfile
        pidfile = tmp_path / "pidfile.pid"
        pidfile.write_text(str(sleeper.pid))
        env = os.environ.copy()
        env["PTLIQ_SERVE_PIDFILE"] = str(pidfile)

        # Stop should send SIGTERM and remove the pidfile
        res = runner.invoke(serve_app, ["stop", "--timeout", "2"], env=env)
        assert res.exit_code == 0, res.stdout

        # Process should be terminated shortly after
        t0 = time.time()
        while time.time() - t0 < 3:
            if sleeper.poll() is not None:
                break
            time.sleep(0.05)
        assert sleeper.poll() is not None  # process exited
        assert not pidfile.exists()
    finally:
        # In case it is still alive for some reason, kill it hard
        if sleeper.poll() is None:
            try:
                sleeper.kill()
            except Exception:
                pass
