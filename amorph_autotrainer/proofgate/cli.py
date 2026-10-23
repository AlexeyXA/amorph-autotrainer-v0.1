"""JSON command-line interface for the Proof Gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .core import evaluate, write_report


def _load_input(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object")
    for required in ("y_gate", "pred_baseline"):
        if required not in payload:
            raise ValueError(f"input JSON is missing required key: {required}")
    unknown = sorted(set(payload) - {"y_gate", "pred_baseline", "pred_candidate", "criteria", "metadata"})
    if unknown:
        raise ValueError(f"unknown top-level input key(s): {unknown}")
    criteria = payload.get("criteria", {})
    if not isinstance(criteria, dict):
        raise ValueError("criteria must be a JSON object")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="proofgate",
        description="Evaluate precomputed baseline/candidate predictions and write a replay-oriented report.json.",
    )
    parser.add_argument("--input", required=True, help="JSON containing y_gate, pred_baseline, optional pred_candidate and criteria")
    parser.add_argument("--output", default="report.json", help="Output path for report.json")
    args = parser.parse_args(argv)

    try:
        payload = _load_input(Path(args.input))
        criteria = payload.get("criteria", {})
        report = evaluate(
            payload["y_gate"],
            payload["pred_baseline"],
            payload.get("pred_candidate"),
            metadata=payload.get("metadata", {}),
            **criteria,
        )
        output = write_report(report, args.output)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"proofgate: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "verdict": output["verdict"],
        "subtype": output["subtype"],
        "report_sha256": output["report_sha256"],
        "output": str(Path(args.output)),
    }, ensure_ascii=False))
    # Exit 0 means a report was produced. The verdict is data, not a CLI execution failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
