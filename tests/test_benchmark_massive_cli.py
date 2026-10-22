from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_massive_ru import requested_strategies


def test_requested_strategy_order_is_stable_and_keeps_tfidf_reference():
    assert requested_strategies(False, False) == ["tfidf"]
    assert requested_strategies(True, False) == ["tfidf", "e5"]
    assert requested_strategies(False, True) == ["tfidf", "hybrid_tfidf_e5"]
    assert requested_strategies(True, True) == ["tfidf", "e5", "hybrid_tfidf_e5"]


def test_benchmark_cli_exposes_hybrid_experiment_flags():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "benchmark_massive_ru.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for flag in (
        "--with-hybrid",
        "--hybrid-classifier-c",
        "--hybrid-class-weight",
        "--e5-classifier-c",
        "--e5-class-weight",
    ):
        assert flag in result.stdout
