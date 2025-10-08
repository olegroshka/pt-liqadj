# ptliq/model/__init__.py

from __future__ import annotations

# --- Components ---
from .backbone import BondBackbone, NodeFieldSpec
from .portfolio_encoder import PortfolioEncoder

# Backward-compat alias: some code imports BondGraphBackbone
BondGraphBackbone = BondBackbone

# Baseline MLP used elsewhere
from .baseline import MLPRegressor  # noqa: F401

# Utility helpers used by training code
from .utils import resolve_device  # re-export for gnn_loop

# --- Optional full model + config ---
# Import lazily so component-only tests still import cleanly even if model.py
# is mid-iteration.
try:
    from .model import PortfolioResidualModel, ModelConfig  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    class _MissingSymbol:  # pragma: no cover
        def __init__(self, name: str):
            self._name = name
        def __getattr__(self, _):
            raise ImportError(
                f"`{self._name}` is not available in ptliq.model. "
                f"Ensure ptliq/model/model.py defines `{self._name}` "
                f"and your install is up to date."
            )
    PortfolioResidualModel = _MissingSymbol("PortfolioResidualModel")  # type: ignore
    ModelConfig = _MissingSymbol("ModelConfig")  # type: ignore

__all__ = [
    # components
    "BondBackbone",
    "BondGraphBackbone",
    "NodeFieldSpec",
    "PortfolioEncoder",
    # utilities
    "resolve_device",
    # baseline
    "MLPRegressor",
    # full model (if present)
    "PortfolioResidualModel",
    "ModelConfig",
]
