from scripts.benchmark_yandex_geo_reviews import organization_group_id, prepare_review_records, rating_to_label


def test_rating_mapping_excludes_zero():
    assert [rating_to_label(x) for x in range(6)] == [
        None, "negative", "negative", "neutral", "positive", "positive"
    ]


def test_group_id_is_stable_and_normalized():
    assert organization_group_id("Moscow, Street 1", " Shop A ") == organization_group_id(
        "moscow, street   1", "shop a"
    )
    assert len(organization_group_id("Moscow, Street 1", "Shop A")) == 24


def _rows():
    rows = []
    for label, rating in (("negative", 1), ("neutral", 3), ("positive", 5)):
        for org in range(12):
            for review in range(3):
                rows.append({
                    "text": f"Distinct review {label} organization {org} item {review}",
                    "rating": rating,
                    "address": f"Street {org}, city {label}",
                    "name_ru": f"Organization {org}",
                })
    return rows


def test_sample_is_deterministic_and_audit_has_no_raw_org_metadata():
    first, audit1 = prepare_review_records(_rows(), max_rows=100)
    second, audit2 = prepare_review_records(_rows(), max_rows=100)
    assert first == second
    assert audit1 == audit2
    assert len(first) <= 100
    assert all(set(row) == {"text", "label", "group_id"} for row in first)
    assert audit1["raw_text_in_audit"] is False
    assert audit1["organization_names_or_addresses_in_audit"] is False


def test_conflicting_duplicate_text_is_removed():
    rows = _rows() + [
        {"text": "Repeated exact review", "rating": 1, "address": "A", "name_ru": "A"},
        {"text": " repeated   exact REVIEW ", "rating": 5, "address": "B", "name_ru": "B"},
    ]
    prepared, audit = prepare_review_records(rows, max_rows=100)
    assert all("repeated exact review" not in " ".join(r["text"].casefold().split()) for r in prepared)
    assert audit["conflicting_normalized_text_keys"] == 1
    assert audit["rows_removed_for_conflicting_exact_text_labels"] == 2
