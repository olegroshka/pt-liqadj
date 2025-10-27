from __future__ import annotations
import os
from pathlib import Path
import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_cli_defaults(tmp_path_factory: pytest.TempPathFactory):
    """
    Ensure tests never write into the real project data/models directories by
    overriding CLI default locations via environment variables for the duration
    of the test session. This keeps unit/integration tests isolated from manual
    CLI runs and local artifacts.
    """
    base = tmp_path_factory.mktemp("ptliq_test_isolation")
    graph_dir = base / "graph"
    pyg_dir = base / "pyg"
    models_dir = base / "models"

    graph_dir.mkdir(parents=True, exist_ok=True)
    pyg_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Point CLI defaults to temp dirs
    os.environ.setdefault("PTLIQ_TEST_MODE", "1")
    os.environ.setdefault("PTLIQ_DEFAULT_GRAPH_DIR", str(graph_dir))
    os.environ.setdefault("PTLIQ_DEFAULT_PYG_DIR", str(pyg_dir))
    os.environ.setdefault("PTLIQ_DEFAULT_MODELS_DIR", str(models_dir))

    # Trades default: prefer simulated fixture if present (read-only), else create a tiny placeholder
    default_trades = Path("data/raw/sim/trades.parquet")
    if default_trades.exists():
        os.environ.setdefault("PTLIQ_DEFAULT_TRADES_PATH", str(default_trades))
    else:
        # Create a minimal empty parquet to satisfy path existence if something relies on default
        try:
            import pandas as pd
            df = pd.DataFrame({"isin": [], "price": [], "trade_dt": []})
            placeholder = base / "trades.parquet"
            df.to_parquet(placeholder, index=False)
            os.environ.setdefault("PTLIQ_DEFAULT_TRADES_PATH", str(placeholder))
        except Exception:
            # If pandas/pyarrow not available in this environment, skip creating the placeholder
            pass

    # Optionally isolate current working directory for tests that use relative paths implicitly
    # Uncomment if needed:
    # os.chdir(base)

    yield

    # No teardown necessary; tmp dirs will be removed by pytest automatically
