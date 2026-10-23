from __future__ import annotations

from scripts.benchmark_public_runnlu_intent_ru import prepare_records


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
    assert audit["rows_retained_before_AMORPH_deduplication"] == 2
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
