from __future__ import annotations

from scripts.benchmark_public_runnlu_intent_ru import (
    filter_low_support_classes,
    prepare_records,
)


def test_prepare_records_drops_all_conflicting_normalized_text_rows():
    rows = [
        {"text": " Turn on the lights ", "label": "iot__on"},
        {"text": "Turn   on the lights", "label": "general__command"},
        {"text": "What is the weather?", "label": "weather__query"},
        {"text": "What is the weather?", "label": "weather__query"},
    ]

    kept, audit = prepare_records(rows)

    assert kept == [
        {"text": "What is the weather?", "label": "weather__query"},
        {"text": "What is the weather?", "label": "weather__query"},
    ]
    assert audit["input_rows"] == 4
    assert audit["ambiguous_normalized_text_keys_removed"] == 1
    assert audit["rows_removed_for_conflicting_labels"] == 2
    assert audit["rows_retained_before_class_support_filter_and_AMORPH_deduplication"] == 2
    assert len(audit["conflicting_text_keys_sha256"]) == 64
    assert "Turn on the lights" not in str(audit)


def test_prepare_records_drops_blank_rows_without_silently_resolving_labels():
    rows = [
        {"text": "  ", "label": "intent_a"},
        {"text": "hello", "label": ""},
        {"text": "pay my bill", "label": "billing__pay"},
    ]

    kept, audit = prepare_records(rows)

    assert kept == [{"text": "pay my bill", "label": "billing__pay"}]
    assert audit["rows_with_empty_text_or_label_removed"] == 2
    assert audit["rows_removed_for_conflicting_labels"] == 0
    assert audit["distinct_labels_after_conflict_removal"] == 1


def test_prepare_records_normalizes_whitespace_and_case_only_for_conflict_detection():
    rows = [
        {"text": "Book a taxi", "label": "transport__book"},
        {"text": "book a   taxi", "label": "transport__book"},
        {"text": "Book a taxi ", "label": "other__label"},
    ]

    kept, audit = prepare_records(rows)

    assert kept == []
    assert audit["ambiguous_normalized_text_keys_removed"] == 1
    assert audit["rows_removed_for_conflicting_labels"] == 3


def test_support_filter_excludes_rare_classes_but_does_not_merge_or_relabel_them():
    rows = [
        *[{"text": f"common intent utterance {i}", "label": "common"} for i in range(20)],
        {"text": "rare one", "label": "rare"},
        {"text": "rare two", "label": "rare"},
        {"text": "rare three", "label": "rare"},
    ]

    kept, audit = filter_low_support_classes(rows, minimum_unique_texts_per_label=20)

    assert len(kept) == 20
    assert {row["label"] for row in kept} == {"common"}
    assert audit["excluded_labels_and_unique_text_counts"] == {"rare": 3}
    assert audit["rows_excluded_for_low_class_support"] == 3
    assert audit["minimum_unique_texts_per_label"] == 20
    assert "rare one" not in str(audit)


def test_support_filter_does_not_count_duplicate_rows_as_unique_class_support():
    rows = [
        *[{"text": f"utterance {i}", "label": "common"} for i in range(19)],
        {"text": "utterance 0", "label": "common"},
    ]

    kept, audit = filter_low_support_classes(rows, minimum_unique_texts_per_label=20)

    assert kept == []
    assert audit["excluded_labels_and_unique_text_counts"] == {"common": 19}
    assert audit["rows_excluded_for_low_class_support"] == 20
