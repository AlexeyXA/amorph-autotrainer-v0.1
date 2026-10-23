from __future__ import annotations

from scripts.benchmark_public_customer_intents import (
    MAX_AUDIT_ROWS,
    canonical_rows_sha256,
    prepare_training_rows,
    resolve_label,
    exclude_exact_test_text_overlaps,
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

def test_clinc_missing_classlabel_is_explicitly_mapped_to_out_of_scope():
    assert resolve_label(None, None, missing_label="out_of_scope") == "out_of_scope"


def test_missing_labels_fail_closed_without_a_predeclared_oos_mapping():
    try:
        resolve_label(None, None)
    except ValueError as exc:
        assert "Missing label" in str(exc)
    else:
        raise AssertionError("Missing labels must fail closed without an explicit mapping")


def test_exact_official_test_text_is_excluded_before_fit_without_using_labels():
    train = [
        {"text": "same text", "label": "alpha"},
        {"text": "  SAME   TEXT  ", "label": "alpha"},
        {"text": "training only 1", "label": "alpha"},
        {"text": "training only 2", "label": "beta"},
    ]
    cleaned, audit = exclude_exact_test_text_overlaps(
        train, ["Same text", "external test only"]
    )
    assert [row["text"] for row in cleaned] == ["training only 1", "training only 2"]
    assert audit["training_rows_removed_for_exact_normalized_test_text_overlap"] == 2
    assert audit["overlapping_normalized_text_keys"] == 1
    assert audit["test_labels_read_during_hygiene"] is False
    assert "same text" not in str(audit).casefold()
