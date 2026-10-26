from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class CostConfig:
    max_runtime_seconds: float = 7200.0
    max_candidates: int = 3
    expected_queries: int = 1000
    compute_eur_per_hour: float = 0.0
    fixed_data_cost_eur: float = 0.0
    maintenance_cost_eur: float = 0.0
    human_hours: float = 0.0
    human_eur_per_hour: float = 0.0
    noninferiority_margin: float = 0.01
    min_macro_f1: float | None = None
    min_recall_by_label: dict[str, float] | None = None
    min_precision_by_label: dict[str, float] | None = None
    bootstrap_samples: int = 500
    min_gate_examples_per_class: int = 5
    random_seed: int = 42

    def validate(self) -> None:
        finite_values = {
            "max_runtime_seconds": self.max_runtime_seconds,
            "expected_queries": self.expected_queries,
            "compute_eur_per_hour": self.compute_eur_per_hour,
            "fixed_data_cost_eur": self.fixed_data_cost_eur,
            "maintenance_cost_eur": self.maintenance_cost_eur,
            "human_hours": self.human_hours,
            "human_eur_per_hour": self.human_eur_per_hour,
            "noninferiority_margin": self.noninferiority_margin,
        }
        if self.min_macro_f1 is not None:
            finite_values["min_macro_f1"] = self.min_macro_f1
        for mapping_name, mapping in (
            ("min_recall_by_label", self.min_recall_by_label),
            ("min_precision_by_label", self.min_precision_by_label),
        ):
            if mapping is None:
                continue
            if not isinstance(mapping, dict):
                raise ValueError(f"{mapping_name} must be a mapping of label to threshold")
            for label, threshold in mapping.items():
                if not isinstance(label, str) or not label.strip():
                    raise ValueError(f"{mapping_name} labels must be non-empty strings")
                if (
                    not isinstance(threshold, (int, float))
                    or not math.isfinite(float(threshold))
                    or not 0 <= float(threshold) <= 1
                ):
                    raise ValueError(f"{mapping_name}[{label!r}] must be finite and in [0, 1]")
        for name, value in finite_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be > 0")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.expected_queries < 0:
            raise ValueError("expected_queries must be >= 0")
        numeric_nonnegative = {
            "compute_eur_per_hour": self.compute_eur_per_hour,
            "fixed_data_cost_eur": self.fixed_data_cost_eur,
            "maintenance_cost_eur": self.maintenance_cost_eur,
            "human_hours": self.human_hours,
            "human_eur_per_hour": self.human_eur_per_hour,
        }
        for name, value in numeric_nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0 <= self.noninferiority_margin < 1:
            raise ValueError("noninferiority_margin must be in [0, 1)")
        if self.min_macro_f1 is not None and not 0 <= self.min_macro_f1 <= 1:
            raise ValueError("min_macro_f1 must be in [0, 1]")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be >= 100")
        if self.min_gate_examples_per_class < 2:
            raise ValueError("min_gate_examples_per_class must be >= 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_tco_eur(
    *,
    fit_seconds: float,
    inference_seconds_per_query: float,
    config: CostConfig,
) -> dict[str, float | str]:
    train_hours = max(0.0, fit_seconds) / 3600.0
    inference_hours = max(0.0, inference_seconds_per_query) * config.expected_queries / 3600.0
    compute_eur = (train_hours + inference_hours) * config.compute_eur_per_hour
    human_eur = config.human_hours * config.human_eur_per_hour
    total = compute_eur + config.fixed_data_cost_eur + config.maintenance_cost_eur + human_eur
    # Runtime proxy remains useful when free compute has a zero direct euro rate.
    runtime_proxy_seconds = max(0.0, fit_seconds) + max(0.0, inference_seconds_per_query) * config.expected_queries
    return {
        "estimated_tco_eur": float(total),
        "compute_cost_eur": float(compute_eur),
        "human_cost_eur": float(human_eur),
        "fixed_data_cost_eur": float(config.fixed_data_cost_eur),
        "maintenance_cost_eur": float(config.maintenance_cost_eur),
        "runtime_proxy_seconds": float(runtime_proxy_seconds),
        "cost_interpretation": (
            "Estimate only. Zero compute rate means compute is assumed free; it does not imply zero economic cost."
        ),
    }
