from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class Record:
    text: str
    label: str
    row_number: int
    group_id: str | None = None


@dataclass
class DatasetBundle:
    train_texts: list[str]
    train_labels: np.ndarray
    val_texts: list[str]
    val_labels: np.ndarray
    gate_texts: list[str]
    gate_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray
    dataset_sha256: str
    report: dict[str, Any]


def normalize_for_dedup(text: str) -> str:
    """Conservative exact duplicate key; does not remove near-duplicates."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_hash(texts: list[str], labels: np.ndarray) -> str:
    canonical = "\n".join(f"{label}\t{normalize_for_dedup(text)}" for text, label in zip(texts, labels))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_records(
    csv_path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    group_column: str | None = None,
) -> tuple[list[Record], dict[str, Any], str]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    source_hash = file_sha256(path)
    records: list[Record] = []
    dropped_empty = 0
    dropped_duplicates = 0
    seen: dict[str, Record] = {}
    conflicts: list[dict[str, Any]] = []
    input_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if text_column not in headers or label_column not in headers:
            raise ValueError(
                f"Required columns not found. Expected text='{text_column}' and label='{label_column}'. "
                f"Available columns: {headers}"
            )
        if group_column is not None:
            if group_column in {text_column, label_column}:
                raise ValueError("group_column must be different from text_column and label_column")
            if group_column not in headers:
                raise ValueError(
                    f"Group column '{group_column}' not present in {path}. "
                    f"Available columns: {headers}"
                )

        for row_number, row in enumerate(reader, start=2):
            input_rows += 1
            text = (row.get(text_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            group_id = (row.get(group_column) or "").strip() if group_column else None
            if not text or not label:
                dropped_empty += 1
                continue
            if group_column is not None and not group_id:
                raise ValueError(
                    f"Missing group value in column '{group_column}' at CSV row {row_number}. "
                    "Group-based splitting fails closed; provide a group for every non-empty record."
                )
            key = normalize_for_dedup(text)
            if key in seen:
                old = seen[key]
                if old.label != label:
                    conflicts.append({
                        "first_row": old.row_number,
                        "conflicting_row": row_number,
                        "first_label": old.label,
                        "conflicting_label": label,
                    })
                else:
                    dropped_duplicates += 1
                continue
            records.append(Record(text=text, label=label, row_number=row_number, group_id=group_id))
            seen[key] = records[-1]

    if conflicts:
        preview = json.dumps(conflicts[:5], ensure_ascii=False)
        raise ValueError(
            "Conflicting labels detected for identical normalized texts. "
            f"Fix labels before training. Examples: {preview}"
        )

    class_counts: dict[str, int] = {}
    for record in records:
        class_counts[record.label] = class_counts.get(record.label, 0) + 1

    if len(class_counts) < 2:
        raise ValueError("Classification requires at least two distinct labels.")
    if min(class_counts.values()) < 5:
        raise ValueError(
            "Each class needs at least 5 unique examples for the default train/validation/test split. "
            f"Counts: {class_counts}"
        )

    report = {
        "source_file": path.name,
        "source_sha256": source_hash,
        "input_rows": input_rows,
        "unique_rows": len(records),
        "dropped_empty_rows": dropped_empty,
        "dropped_exact_duplicates_same_label": dropped_duplicates,
        "conflicting_exact_duplicates": 0,
        "class_counts": dict(sorted(class_counts.items())),
        "num_classes": len(class_counts),
        "group_column": group_column,
        "distinct_external_groups": (
            len({record.group_id for record in records})
            if group_column is not None else None
        ),
    }
    return records, report, source_hash


def _near_duplicate_groups(
    records: list[Record],
    *,
    threshold: float,
    max_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bounded char-ngram cosine audit; near-duplicates are grouped across splits.

    This is a conservative engineering heuristic, not semantic duplicate proof.
    It is bounded because exact nearest-neighbour search can approach O(N^2).
    """
    n = len(records)
    if n > max_rows:
        return np.arange(n, dtype=np.int64), {
            "status": "skipped_size_limit",
            "max_rows": max_rows,
            "rows": n,
            "warning": "Near-duplicate audit and grouped split were skipped above the configured limit.",
            "threshold": threshold,
            "pairs": [],
        }

    texts = [record.text for record in records]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1,
        max_features=200_000, dtype=np.float32, norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    if n == 1:
        return np.zeros(n, dtype=np.int64), {"status": "completed", "threshold": threshold, "pairs": []}

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1)
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix, n_neighbors=min(6, n))

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for i, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for dist, j in zip(row_distances, row_indices):
            j = int(j)
            if j == i:
                continue
            similarity = 1.0 - float(dist)
            if similarity >= threshold:
                left, right = sorted((i, j))
                union(left, right)
                pairs[(left, right)] = {
                    "row_a": records[left].row_number,
                    "row_b": records[right].row_number,
                    "similarity": round(similarity, 5),
                    "same_label": records[left].label == records[right].label,
                }

    roots = [find(i) for i in range(n)]
    remap: dict[int, int] = {}
    group_ids = np.asarray([remap.setdefault(root, len(remap)) for root in roots], dtype=np.int64)
    pair_list = sorted(pairs.values(), key=lambda item: item["similarity"], reverse=True)
    mixed_label_pairs = sum(1 for pair in pair_list if not pair["same_label"])
    report = {
        "status": "completed",
        "method": "char_wb_tfidf_cosine_nearest_5",
        "threshold": threshold,
        "rows": n,
        "groups_after_near_duplicate_clustering": int(len(set(group_ids.tolist()))),
        "near_duplicate_pairs_found": len(pair_list),
        "near_duplicate_pairs_with_different_labels": mixed_label_pairs,
        "pairs": pair_list[:100],
        "warning": "Similarity threshold is heuristic; review cross-label matches and use domain-specific grouping for time/user/source data.",
    }
    return group_ids, report


def _choose_four_way_group_split(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    train_fraction: float,
    validation_fraction: float,
    gate_fraction: float,
    test_fraction: float,
    time_limit_seconds: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Assign whole groups to train/validation/gate/test jointly with a MILP.

    The optimizer must put every class in every split and each group in exactly one
    split. Its objective minimizes normalized absolute deviations from requested row
    fractions. A time-limited incumbent is accepted only after validating all hard
    constraints; otherwise this routine fails closed.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    split_names = ("train", "validation", "acceptance_gate", "test")
    fractions = np.asarray(
        [train_fraction, validation_fraction, gate_fraction, test_fraction],
        dtype=float,
    )
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0, atol=1e-8):
        raise ValueError(f"Invalid grouped split fractions: {fractions.tolist()}")

    subset_labels = labels[indices]
    subset_groups = groups[indices]
    unique_groups, group_inverse = np.unique(subset_groups, return_inverse=True)
    unique_labels = sorted(set(subset_labels.tolist()), key=str)
    group_count = len(unique_groups)
    label_count = len(unique_labels)
    split_count = len(split_names)
    if group_count < split_count:
        raise ValueError(
            f"At least {split_count} distinct groups are required for four grouped splits; "
            f"received {group_count}."
        )

    label_to_column = {label: column for column, label in enumerate(unique_labels)}
    group_has_label = np.zeros((group_count, label_count), dtype=bool)
    for group_index, label in zip(group_inverse, subset_labels):
        group_has_label[group_index, label_to_column[label]] = True
    group_weights = np.bincount(group_inverse, minlength=group_count).astype(float)

    # Binary x[group, split] variables, followed by one continuous absolute-size
    # deviation variable per split.
    binary_variable_count = group_count * split_count
    variable_count = binary_variable_count + split_count
    deviation_start = binary_variable_count
    total_rows = len(indices)
    constraint_count = group_count + label_count * split_count + 2 * split_count
    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    lower_constraints: list[float] = []
    upper_constraints: list[float] = []
    constraint_row = 0

    # Every source group is assigned to exactly one split.
    for group_index in range(group_count):
        for split_index in range(split_count):
            row_indices.append(constraint_row)
            column_indices.append(group_index * split_count + split_index)
            coefficients.append(1.0)
        lower_constraints.append(1.0)
        upper_constraints.append(1.0)
        constraint_row += 1

    # Every label must appear in all four splits.
    for label_column in range(label_count):
        member_groups = np.flatnonzero(group_has_label[:, label_column])
        for split_index in range(split_count):
            for group_index in member_groups:
                row_indices.append(constraint_row)
                column_indices.append(int(group_index) * split_count + split_index)
                coefficients.append(1.0)
            lower_constraints.append(1.0)
            upper_constraints.append(np.inf)
            constraint_row += 1

    # Minimize absolute row-count deviation from each target fraction.
    for split_index in range(split_count):
        deviation_column = deviation_start + split_index
        for group_index, weight in enumerate(group_weights):
            row_indices.append(constraint_row)
            column_indices.append(group_index * split_count + split_index)
            coefficients.append(float(weight))
        row_indices.append(constraint_row)
        column_indices.append(deviation_column)
        coefficients.append(-1.0)
        lower_constraints.append(-np.inf)
        upper_constraints.append(float(fractions[split_index] * total_rows))
        constraint_row += 1

        for group_index, weight in enumerate(group_weights):
            row_indices.append(constraint_row)
            column_indices.append(group_index * split_count + split_index)
            coefficients.append(-float(weight))
        row_indices.append(constraint_row)
        column_indices.append(deviation_column)
        coefficients.append(-1.0)
        lower_constraints.append(-np.inf)
        upper_constraints.append(-float(fractions[split_index] * total_rows))
        constraint_row += 1

    constraints_matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(constraint_count, variable_count),
        dtype=float,
    ).tocsr()
    objective = np.zeros(variable_count, dtype=float)
    objective[deviation_start:] = 1.0 / fractions
    integrality = np.zeros(variable_count, dtype=int)
    integrality[:binary_variable_count] = 1
    variable_lower = np.zeros(variable_count, dtype=float)
    variable_upper = np.concatenate(
        [np.ones(binary_variable_count, dtype=float), np.full(split_count, np.inf)]
    )

    started = time.perf_counter()
    solution = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(
            constraints_matrix,
            np.asarray(lower_constraints, dtype=float),
            np.asarray(upper_constraints, dtype=float),
        ),
        options={
            "time_limit": time_limit_seconds,
            "presolve": True,
            "mip_rel_gap": 0.01,
        },
    )
    solve_seconds = time.perf_counter() - started
    if solution.x is None:
        raise ValueError(
            "Could not construct a leakage-safe four-way grouped split with every class "
            f"represented in every split within {time_limit_seconds:.1f}s. "
            f"Optimizer status={solution.status}: {solution.message}. "
            "Collect more independent groups or revise the predeclared split policy."
        )

    allocation = solution.x[:binary_variable_count].reshape(group_count, split_count)
    if not np.allclose(allocation.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Group-split optimizer returned an invalid group assignment.")
    group_assignment = np.argmax(allocation, axis=1)
    missing_coverage = {
        str(label): [
            split_names[split_index]
            for split_index in range(split_count)
            if not np.any(
                group_has_label[:, label_column] & (group_assignment == split_index)
            )
        ]
        for label_column, label in enumerate(unique_labels)
    }
    missing_coverage = {
        label: missing_splits
        for label, missing_splits in missing_coverage.items()
        if missing_splits
    }
    if missing_coverage:
        raise RuntimeError(
            "Group-split optimizer returned a partition with missing classes: "
            + json.dumps(missing_coverage, ensure_ascii=False)
        )

    row_assignment = group_assignment[group_inverse]
    partition_indices = [
        indices[row_assignment == split_index]
        for split_index in range(split_count)
    ]
    actual_rows = {
        split_names[split_index]: int(len(partition_indices[split_index]))
        for split_index in range(split_count)
    }
    actual_groups = {
        split_names[split_index]: int(np.sum(group_assignment == split_index))
        for split_index in range(split_count)
    }
    actual_fractions = {
        name: actual_rows[name] / total_rows for name in split_names
    }
    status_names = {
        0: "optimal",
        1: "feasible_incumbent_at_time_limit",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }
    metadata = {
        "solver": "scipy.optimize.milp (HiGHS)",
        "solver_status": status_names.get(int(solution.status), f"status_{solution.status}"),
        "solver_message": str(solution.message),
        "objective_value": (
            float(solution.fun) if solution.fun is not None else None
        ),
        "time_limit_seconds": float(time_limit_seconds),
        "solve_seconds": round(solve_seconds, 4),
        "random_seed_used": False,
        "hard_constraints": [
            "each worker/near-duplicate group assigned to exactly one split",
            "every effective training label represented in train, validation, acceptance_gate and test",
        ],
        "target_fractions": {
            split_names[index]: float(fractions[index]) for index in range(split_count)
        },
        "actual_rows": actual_rows,
        "actual_fractions": actual_fractions,
        "groups_by_split": actual_groups,
        "classes_by_split": {
            split_names[index]: label_count for index in range(split_count)
        },
    }
    return (
        partition_indices[0],
        partition_indices[1],
        partition_indices[2],
        partition_indices[3],
        metadata,
    )


def _merge_group_ids(
    near_duplicate_groups: np.ndarray,
    external_groups: list[str | None],
) -> np.ndarray:
    """Combine detected near-duplicate clusters with user-specified group IDs.

    Rows are in the same final group if they are connected by either signal.
    When the near-duplicate audit is skipped, its IDs are row-unique; external
    groups still provide a valid group-safe split.
    """
    n = len(near_duplicate_groups)
    if len(external_groups) != n:
        raise ValueError("External group IDs must have the same length as the dataset")

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    def union_equal_keys(keys: list[Any]) -> None:
        first_by_key: dict[Any, int] = {}
        for idx, key in enumerate(keys):
            if key in first_by_key:
                union(first_by_key[key], idx)
            else:
                first_by_key[key] = idx

    union_equal_keys(near_duplicate_groups.tolist())
    nonempty_external = [
        (value if value else None) for value in external_groups
    ]
    if any(value is not None for value in nonempty_external):
        # Missing external group IDs are disallowed by load_records when a group
        # column is enabled. Keep None distinct here for direct API callers.
        union_equal_keys([
            ("external", value) if value is not None else ("row", idx)
            for idx, value in enumerate(nonempty_external)
        ])

    roots = [find(i) for i in range(n)]
    remap: dict[int, int] = {}
    return np.asarray([remap.setdefault(root, len(remap)) for root in roots], dtype=np.int64)


def split_records(
    records: list[Record],
    dataset_sha256: str,
    initial_report: dict[str, Any],
    seed: int = 42,
    test_size: float = 0.20,
    validation_size: float = 0.15,
    gate_size: float = 0.15,
    near_duplicate_threshold: float = 0.985,
    max_near_duplicate_audit_rows: int = 5000,
) -> DatasetBundle:
    if (not 0 < test_size < 0.5 or not 0 < validation_size < 0.5
            or not 0 < gate_size < 0.5 or test_size + validation_size + gate_size >= 0.8):
        raise ValueError("test_size, validation_size and gate_size must be positive and sum to less than 0.8")
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")

    texts = np.asarray([item.text for item in records], dtype=object)
    labels = np.asarray([item.label for item in records], dtype=object)
    indices = np.arange(len(records))
    groups, duplicate_report = _near_duplicate_groups(
        records, threshold=near_duplicate_threshold, max_rows=max_near_duplicate_audit_rows
    )
    external_group_values = [record.group_id for record in records]
    has_external_groups = any(value is not None for value in external_group_values)
    if has_external_groups:
        groups = _merge_group_ids(groups, external_group_values)

    group_counts_by_class: dict[str, int] = {}
    for label in sorted(set(labels.tolist())):
        group_counts_by_class[label] = len(set(groups[labels == label].tolist()))

    near_duplicate_grouping_used = duplicate_report.get("status") == "completed"
    group_split_used = near_duplicate_grouping_used or has_external_groups
    group_split_optimizer_metadata = None
    if group_split_used:
        too_few = {label: count for label, count in group_counts_by_class.items() if count < 5}
        if too_few:
            raise ValueError(
                "Near-duplicate grouping leaves fewer than 5 distinct groups in some classes, "
                f"so an honest train/validation/gate/test split is not reliable: {too_few}. "
                "Collect more examples or review/adjust the near-duplicate threshold."
            )

        train_fraction = 1.0 - test_size - validation_size - gate_size
        (
            train_idx,
            val_idx,
            gate_idx,
            test_idx,
            group_split_optimizer_metadata,
        ) = _choose_four_way_group_split(
            indices,
            labels,
            groups,
            train_fraction=train_fraction,
            validation_fraction=validation_size,
            gate_fraction=gate_size,
            test_fraction=test_size,
        )
    else:
        # Explicit fallback: exact duplicates were removed globally; near-duplicate
        # and domain-specific leakage guarantees are unavailable above the size cap.
        trainvalgate_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=labels
        )
        gate_fraction_of_remaining = gate_size / (1.0 - test_size)
        trainval_idx, gate_idx = train_test_split(
            trainvalgate_idx, test_size=gate_fraction_of_remaining,
            random_state=seed + 1, stratify=labels[trainvalgate_idx]
        )
        val_fraction_of_remaining = validation_size / (1.0 - test_size - gate_size)
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=val_fraction_of_remaining,
            random_state=seed + 2, stratify=labels[trainval_idx]
        )

    def extract(idx: np.ndarray) -> tuple[list[str], np.ndarray]:
        return texts[idx].tolist(), labels[idx]

    train_texts, train_labels = extract(train_idx)
    val_texts, val_labels = extract(val_idx)
    gate_texts, gate_labels = extract(gate_idx)
    test_texts, test_labels = extract(test_idx)

    # Never silently split members of the same detected duplicate group.
    split_indices = (train_idx, val_idx, gate_idx, test_idx)
    split_group_sets = [set(groups[idx].tolist()) for idx in split_indices]
    for i in range(len(split_group_sets)):
        for j in range(i + 1, len(split_group_sets)):
            if split_group_sets[i] & split_group_sets[j]:
                raise RuntimeError("Data leakage: a detected near-duplicate group spans multiple splits.")

    report = dict(initial_report)
    report["near_duplicate_audit"] = duplicate_report
    report["group_counts_by_class"] = group_counts_by_class
    if has_external_groups and near_duplicate_grouping_used:
        split_method = "milp_group_assignment_with_group_column_and_near_duplicates"
    elif has_external_groups:
        split_method = "milp_group_assignment_with_group_column_near_duplicate_audit_skipped"
    elif near_duplicate_grouping_used:
        split_method = "milp_group_assignment_with_near_duplicate_groups"
    else:
        split_method = "stratified_exact_dedup_fallback"
    report["split_method"] = split_method
    if group_split_optimizer_metadata is not None:
        report["group_split_optimizer"] = group_split_optimizer_metadata
    report["group_column_used"] = initial_report.get("group_column")
    report["split_purpose"] = {
        "train": "fit model parameters",
        "validation": "select strategy/hyperparameters",
        "acceptance_gate": "paired bootstrap non-inferiority and predeclared quality target",
        "test": "final reporting only; not used to select the model",
    }
    report["splits"] = {
        "train": {"rows": len(train_texts), "sha256": _split_hash(train_texts, train_labels)},
        "validation": {"rows": len(val_texts), "sha256": _split_hash(val_texts, val_labels)},
        "acceptance_gate": {"rows": len(gate_texts), "sha256": _split_hash(gate_texts, gate_labels)},
        "test": {"rows": len(test_texts), "sha256": _split_hash(test_texts, test_labels)},
    }
    report["split_seed"] = seed

    return DatasetBundle(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        gate_texts=gate_texts,
        gate_labels=gate_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        dataset_sha256=dataset_sha256,
        report=report,
    )

