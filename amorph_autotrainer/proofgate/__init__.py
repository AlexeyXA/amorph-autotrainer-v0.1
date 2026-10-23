"""Small, dependency-light quality gate API for precomputed classification predictions."""

from .core import GateReport, Verdict, evaluate, write_report

__all__ = ["GateReport", "Verdict", "evaluate", "write_report"]
