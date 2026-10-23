from __future__ import annotations

from scripts.benchmark_grouped_author_profiling_ru import prepare_task_records


def test_filters_not_applicable_rows_and_keeps_group_metadata():
    rows = [
        {"text": "A text", "author_id": 1, "age_imitation": "younger"},
        {"text": "B text", "author_id": 1, "age_imitation": "older"},
        {"text": "C text", "author_id": 1, "age_imitation": "no_age_imitation"},
        {"text": "not in this task", "author_id": 1, "age_imitation": "None"},
    ]

    kept, audit = prepare_task_records(rows)

    assert [row["label"] for row in kept] == ["younger", "older", "no_age_imitation"]
    assert {row["author_id"] for row in kept} == {"1"}
    assert audit["not_applicable_or_unrecognized_task_labels_removed"] == 1
    assert audit["unique_author_groups_retained"] == 1
    assert "A text" not in str(audit)


def test_drops_all_identical_texts_if_the_task_labels_conflict():
    rows = [
        {"text": " A normalized text ", "author_id": 1, "age_imitation": "younger"},
        {"text": "a   normalized text", "author_id": 2, "age_imitation": "older"},
        {"text": "a third unique sample", "author_id": 3, "age_imitation": "no_age_imitation"},
    ]

    try:
        kept, audit = prepare_task_records(rows)
    except ValueError as exc:
        # The preparation function is allowed to reject a too-small residual
        # dataset after removing ambiguous rows; the audit itself is not returned.
        assert "Expected the three predeclared task labels" in str(exc)
    else:
        assert all(row["text"].casefold().strip() != "a normalized text" for row in kept)
        assert audit["ambiguous_normalized_text_keys_removed"] == 1
        assert audit["rows_removed_for_conflicting_labels"] == 2


def test_drops_identical_text_repeated_across_author_groups():
    rows = [
        {"text": "shared utterance", "author_id": 10, "age_imitation": "younger"},
        {"text": "shared   utterance", "author_id": 11, "age_imitation": "younger"},
        {"text": "other younger sample", "author_id": 12, "age_imitation": "younger"},
        {"text": "other older sample", "author_id": 13, "age_imitation": "older"},
        {"text": "other no imitation sample", "author_id": 14, "age_imitation": "no_age_imitation"},
    ]

    try:
        kept, audit = prepare_task_records(rows)
    except ValueError as exc:
        # Only one row remains per class, below the support floor.
        assert "Each target class needs at least 20" in str(exc)
    else:
        assert all("shared" not in row["text"] for row in kept)
        assert audit["normalized_text_keys_repeated_across_authors_removed"] == 1
        assert audit["rows_removed_for_cross_author_duplicate_text"] == 2


def test_collapses_exact_duplicate_within_same_author_and_label():
    rows = [
        {"text": "same author sample", "author_id": 7, "age_imitation": "younger"},
        {"text": " SAME   AUTHOR sample ", "author_id": 7, "age_imitation": "younger"},
        {"text": "older sample", "author_id": 8, "age_imitation": "older"},
        {"text": "no imitation sample", "author_id": 9, "age_imitation": "no_age_imitation"},
    ]

    try:
        kept, audit = prepare_task_records(rows)
    except ValueError as exc:
        # The tiny test fixture intentionally fails the production minimum.
        assert "Each target class needs at least 20" in str(exc)
    else:
        assert len([row for row in kept if row["label"] == "younger"]) == 1
        assert audit["rows_removed_for_same_author_same_text_duplicate"] == 1


def test_rejects_missing_group_ids_fail_closed():
    rows = [
        {"text": "sample", "author_id": None, "age_imitation": "younger"},
    ]

    try:
        prepare_task_records(rows)
    except ValueError as exc:
        assert "group-aware evaluation fails closed" in str(exc)
    else:
        raise AssertionError("Missing group IDs must not be silently accepted")
