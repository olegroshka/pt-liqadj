"""
Compatibility shim for scoring import path.

Historically, scorers live under ptliq.service.scoring, but some tests import
`ptliq.scoring`. This module re-exports the public scorer classes so both
import styles work.
"""
from .service.scoring import (
    DGTScorer,
    GRUScorer,
    MLPScorer,
    ModelBundle,
    HasFeatureNames,
    Scorer,
)

__all__ = [
    "DGTScorer",
    "GRUScorer",
    "MLPScorer",
    "ModelBundle",
    "HasFeatureNames",
    "Scorer",
]
