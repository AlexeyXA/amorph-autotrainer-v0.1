import csv
from pathlib import Path

import pytest

from amorph_autotrainer.data import load_records, split_records


def make_csv(path: Path, duplicate_conflict: bool = False):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        for label in ("a", "b", "c"):
            for i in range(10):
                writer.writerow({"text": f"Unique example {label} number {i}", "label": label})
        writer.writerow({"text": "  Unique   example a number 0 ", "label": "a"})
        if duplicate_conflict:
            writer.writerow({"text": "Unique example a number 1", "label": "b"})


def test_exact_duplicates_are_removed_before_split(tmp_path):
    path = tmp_path / "data.csv"
    make_csv(path)
    records, report, digest = load_records(path)
    assert len(records) == 30
    assert report["dropped_exact_duplicates_same_label"] == 1
    bundle = split_records(records, digest, report)
    all_sets = [set(t.casefold() for t in bundle.train_texts),
                set(t.casefold() for t in bundle.val_texts),
                set(t.casefold() for t in bundle.test_texts)]
    assert not all_sets[0] & all_sets[1]
    assert not all_sets[0] & all_sets[2]
    assert not all_sets[1] & all_sets[2]


def test_conflicting_duplicate_labels_fail_closed(tmp_path):
    path = tmp_path / "data.csv"
    make_csv(path, duplicate_conflict=True)
    with pytest.raises(ValueError, match="Conflicting labels"):
        load_records(path)


def test_missing_columns_are_reported(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("body,category\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Required columns"):
        load_records(path)


def test_group_column_prevents_group_leakage_across_all_four_splits(tmp_path):
    path = tmp_path / "grouped.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "account"])
        writer.writeheader()
        # Each account contributes one record to every class; the split must
        # keep all records for one account in a single partition.
        for account in range(10):
            for label in ("a", "b", "c"):
                writer.writerow({
                    "text": f"account-{account:02d} distinctive document class {label} payload {account}-{label}",
                    "label": label,
                    "account": f"account-{account:02d}",
                })

    records, report, digest = load_records(path, group_column="account")
    bundle = split_records(records, digest, report)
    split_texts = [
        bundle.train_texts, bundle.val_texts, bundle.gate_texts, bundle.test_texts,
    ]
    split_groups = [{text.split()[0] for text in texts} for texts in split_texts]
    for i in range(len(split_groups)):
        for j in range(i + 1, len(split_groups)):
            assert not split_groups[i] & split_groups[j]
    effective_labels = set(bundle.train_labels.tolist())
    for split_labels in (
        bundle.train_labels, bundle.val_labels, bundle.gate_labels, bundle.test_labels,
    ):
        assert set(split_labels.tolist()) == effective_labels
    assert bundle.report["split_method"] == "milp_group_assignment_with_group_column_and_near_duplicates"
    assert bundle.report["group_column_used"] == "account"
    assert bundle.report["group_split_optimizer"]["solver"] == "scipy.optimize.milp (HiGHS)"


def test_group_column_requires_a_value_for_each_nonempty_record(tmp_path):
    path = tmp_path / "missing_group.csv"
    path.write_text("text,label,account\nvalid text,a,\nother text,b,account-2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing group value"):
        load_records(path, group_column="account")
