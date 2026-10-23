"""Small, dependency-light quality gate API for precomputed classification predictions."""

from .core import (
    PROOFGATE_VERSION,
    REPORT_SCHEMA_VERSION,
    GateReport,
    Verdict,
    evaluate,
    write_report,
)

__all__ = [
    "PROOFGATE_VERSION",
    "REPORT_SCHEMA_VERSION",
    "GateReport",
    "Verdict",
    "evaluate",
    "write_report",
]
