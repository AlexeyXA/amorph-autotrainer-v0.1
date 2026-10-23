from __future__ import annotations

from scripts.benchmark_grouped_author_profiling_ru import prepare_task_records

LABELS = ("younger", "older", "no_age_imitation")


def supported_rows():
    return [
        {"text": f"{label} base utterance {i}", "author_id": f"{label}_{i}", "age_imitation": label}
        for label in LABELS
        for i in range(20)
    ]


def test_filters_not_applicable_rows_and_keeps_group_metadata():
    rows = supported_rows()
    rows.append({"text": "not in this task", "author_id": 1, "age_imitation": "None"})

    kept, audit = prepare_task_records(rows)

    assert len(kept) == 60
    assert {row["label"] for row in kept} == set(LABELS)
    assert all("author_id" in row for row in kept)
    assert audit["not_applicable_or_unrecognized_task_labels_removed"] == 1
    assert audit["unique_author_groups_retained"] == 60
    assert "younger base utterance" not in str(audit)


def test_drops_all_identical_texts_if_the_task_labels_conflict():
    rows = supported_rows()
    rows.extend([
        {"text": " A normalized text ", "author_id": "conflict_1", "age_imitation": "younger"},
        {"text": "a   normalized text", "author_id": "conflict_2", "age_imitation": "older"},
    ])

    kept, audit = prepare_task_records(rows)

    assert all("normalized text" not in row["text"] for row in kept)
    assert audit["ambiguous_normalized_text_keys_removed"] == 1
    assert audit["rows_removed_for_conflicting_labels"] == 2


def test_drops_identical_text_repeated_across_author_groups():
    rows = supported_rows()
    rows.extend([
        {"text": "shared utterance", "author_id": "author_a", "age_imitation": "younger"},
        {"text": "shared   utterance", "author_id": "author_b", "age_imitation": "younger"},
    ])

    kept, audit = prepare_task_records(rows)

    assert all("shared utterance" not in row["text"] for row in kept)
    assert audit["normalized_text_keys_repeated_across_authors_removed"] == 1
    assert audit["rows_removed_for_cross_author_duplicate_text"] == 2


def test_collapses_exact_duplicate_within_same_author_and_label():
    rows = supported_rows()
    rows.extend([
        {"text": "same author sample", "author_id": "author_same", "age_imitation": "younger"},
        {"text": " SAME   AUTHOR sample ", "author_id": "author_same", "age_imitation": "younger"},
    ])

    kept, audit = prepare_task_records(rows)

    same_author_rows = [row for row in kept if "same author sample" in row["text"].casefold()]
    assert len(same_author_rows) == 1
    assert audit["rows_removed_for_same_author_same_text_duplicate"] == 1


def test_rejects_missing_group_ids_fail_closed():
    rows = supported_rows()
    rows.append({"text": "missing group", "author_id": None, "age_imitation": "younger"})

    try:
        prepare_task_records(rows)
    except ValueError as exc:
        assert "group-aware evaluation fails closed" in str(exc)
    else:
        raise AssertionError("Missing group IDs must not be silently accepted")
