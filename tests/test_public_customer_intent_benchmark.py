from __future__ import annotations

from scripts.benchmark_public_customer_intents import (
    MAX_AUDIT_ROWS,
    canonical_rows_sha256,
    prepare_training_rows,
)


def test_conflicting_exact_text_rows_are_removed_without_relabeling():
    rows = [
        {"text": "Where is my order?", "label": "track_order"},
        {"text": "  where   is my ORDER? ", "label": "cancel_order"},
    ]
    rows.extend(
        {"text": f"unique {i}", "label": "alpha" if i < 30 else "beta"}
        for i in range(60)
    )
    prepared, audit = prepare_training_rows(rows, max_per_class=25)
    assert all("where is my order" not in row["text"].casefold() for row in prepared)
    assert audit["conflicting_normalized_text_keys_removed"] == 1
    assert audit["training_rows_removed_for_conflicting_labels"] == 2
    assert "Where is my order?" not in str(audit)
    assert len(prepared) <= MAX_AUDIT_ROWS


def test_same_label_exact_duplicates_collapse_and_sampling_is_deterministic():
    rows = []
    for label in ("alpha", "beta"):
        for i in range(30):
            rows.append({"text": f"{label} example {i}", "label": label})
    rows += [
        {"text": "alpha example 1", "label": "alpha"},
        {"text": "  ALPHA   EXAMPLE 1  ", "label": "alpha"},
    ]
    one, audit_one = prepare_training_rows(rows, max_per_class=22, seed=42)
    two, audit_two = prepare_training_rows(rows, max_per_class=22, seed=42)
    assert one == two
    assert canonical_rows_sha256(one) == canonical_rows_sha256(two)
    assert len(one) == 44
    assert audit_one["same_label_exact_duplicates_removed"] == 2
    assert audit_one == audit_two


def test_training_cap_never_exceeds_near_duplicate_audit_limit():
    rows = [
        {"text": f"{label} example {i}", "label": label}
        for label in ("alpha", "beta", "gamma")
        for i in range(100)
    ]
    prepared, _ = prepare_training_rows(rows, max_per_class=75)
    assert len(prepared) == 225
    assert len(prepared) <= MAX_AUDIT_ROWS
    assert {row["label"] for row in prepared} == {"alpha", "beta", "gamma"}
