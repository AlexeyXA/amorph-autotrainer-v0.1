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


def test_default_split_fractions_favor_larger_acceptance_gate(tmp_path):
    path = tmp_path / "data.csv"
    make_csv(path)
    records, report, digest = load_records(path)
    bundle = split_records(
        records, digest, report, max_near_duplicate_audit_rows=1,
    )
    n = len(records)
    assert abs(len(bundle.gate_texts) / n - 0.25) < 0.12
    assert abs(len(bundle.train_texts) / n - 0.45) < 0.15


def test_group_split_enforces_min_gate_rows_per_class(tmp_path):
    path = tmp_path / "grouped.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "account"])
        writer.writeheader()
        for account in range(16):
            for label in ("a", "b", "c"):
                writer.writerow({
                    "text": f"acct-{account:02d} class {label} unique body {account}-{label}",
                    "label": label,
                    "account": f"acct-{account:02d}",
                })
    records, report, digest = load_records(path, group_column="account")
    bundle = split_records(
        records, digest, report,
        min_examples_per_label_in_gate=5,
        min_examples_per_label_in_other_splits=1,
    )
    from collections import Counter
    gate_counts = Counter(str(x) for x in bundle.gate_labels.tolist())
    assert min(gate_counts.values()) >= 5
    assert set(gate_counts) == {"a", "b", "c"}
    assert bundle.report["split_method"].startswith("milp_group_assignment")


def test_group_split_fails_closed_when_gate_floor_infeasible(tmp_path):
    path = tmp_path / "tiny_groups.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "account"])
        writer.writeheader()
        for account in range(6):
            for label in ("a", "b"):
                writer.writerow({
                    "text": f"tiny-{account} {label} payload {account}-{label}",
                    "label": label,
                    "account": f"g{account}",
                })
    records, report, digest = load_records(path, group_column="account")
    with pytest.raises(ValueError, match="Could not construct a leakage-safe four-way|underpowered|infeasible|fewer than"):
        split_records(
            records, digest, report,
            min_examples_per_label_in_gate=5,
            min_examples_per_label_in_other_splits=1,
        )
