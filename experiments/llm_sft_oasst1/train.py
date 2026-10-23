#!/usr/bin/env python3
"""Isolated, auditable Russian OASST1 SFT + LoRA proof-of-concept."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from experiments.llm_sft_oasst1.data import encode_split_up_to_cap, prepare_ru_pairs, split_by_tree
from experiments.llm_sft_oasst1.gate import paired_group_bootstrap

DATASET_ID = "OpenAssistant/oasst1"
DATASET_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "ec7ddfa904d4d447eedd0b7f126df16957734abb"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    rows: list[dict[str, Any]] = []
    for split_name in sorted(dataset.keys()):
        rows.extend(dict(row) for row in dataset[split_name])
    return rows, {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "upstream_splits_combined_before_tree_split": sorted(dataset.keys()),
        "upstream_message_rows_loaded": len(rows),
        "license_claim": "Apache-2.0 per the public dataset card; re-check terms before redistribution/commercial use",
    }


def collate(examples: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(examples), width), dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for i, row in enumerate(examples):
        n = len(row["input_ids"])
        input_ids[i, :n] = torch.tensor(row["input_ids"], dtype=torch.long)
        attention[i, :n] = torch.tensor(row["attention_mask"], dtype=torch.long)
        labels[i, :n] = torch.tensor(row["labels"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def score_responses(model: Any, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score one example at a time to bound CPU memory use."""
    device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()
    scored: list[dict[str, Any]] = []
    for row in examples:
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        mask = torch.tensor([row["attention_mask"]], dtype=torch.long, device=device)
        labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].contiguous().view(-1)
            count = int((shift_labels != -100).sum().item())
            if count < 1:
                raise ValueError("An encoded example contains no assistant answer tokens")
            loss_sum = F.cross_entropy(
                shift_logits.float(), shift_labels, ignore_index=-100, reduction="sum"
            ).item()
        scored.append({"tree_id": row["tree_id"], "nll": loss_sum / count, "answer_tokens": count})
        del ids, mask, labels, logits, shift_logits, shift_labels
    if was_training:
        model.train()
    return scored


def mean_nll(rows: list[dict[str, Any]]) -> float:
    # Each independent conversation tree receives equal weight.
    return mean(float(row["nll"]) for row in rows)


def write_inventory(root: Path) -> None:
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "artifact_inventory.json"):
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    write_json(root / "artifact_inventory.json", {"files_sha256": entries, "file_count": len(entries)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-examples", type=int, default=128)
    parser.add_argument("--max-validation-examples", type=int, default=24)
    parser.add_argument("--max-gate-examples", type=int, default=32)
    parser.add_argument("--max-test-examples", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--gate-margin", type=float, default=0.02)
    parser.add_argument("--minimum-gate-groups", type=int, default=20)
    args = parser.parse_args()
    if min(args.max_train_examples, args.max_validation_examples, args.max_gate_examples, args.max_test_examples) < 1:
        parser.error("all split caps must be positive")
    if args.max_steps < 1 or args.max_length < 16 or args.bootstrap_samples < 100:
        parser.error("max-steps>=1, max-length>=16, and bootstrap-samples>=100 are required")

    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading pinned OASST1 and selecting reviewed Russian pairs", flush=True)
    rows, provenance = load_rows()
    pairs, audit = prepare_ru_pairs(rows, seed=args.seed)
    raw_splits = split_by_tree(pairs, seed=args.seed)
    caps = {
        "train": args.max_train_examples,
        "validation": args.max_validation_examples,
        "gate": args.max_gate_examples,
        "test": args.max_test_examples,
    }
    print("[2/7] Loading pinned Qwen model and tokenizer on CPU", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.config.use_cache = False

    print("[3/7] Encoding pairs; prompt tokens are masked from loss", flush=True)
    encoded: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, Any] = {}
    for name, group in raw_splits.items():
        encoded[name], split_stats = encode_split_up_to_cap(
            group, tokenizer, limit=caps[name], max_length=args.max_length,
            seed=args.seed, split_name=name,
        )
        split_counts[name] = split_stats
    if len(encoded["train"]) < 8 or len(encoded["validation"]) < 4 or len(encoded["test"]) < 4:
        raise RuntimeError(f"Insufficient encoded split support: {split_counts}")

    identifiers = sorted(p["tree_id"] + ":" + p["prompt_id"] + ":" + p["reply_id"] for p in pairs)
    write_json(output / "dataset_provenance.json", {
        **provenance,
        "data_audit": audit,
        "split_counts": split_counts,
        "seed": args.seed,
        "max_length": args.max_length,
        "split_policy": "exact normalized prompt dedup across conversation trees; deterministic 70/10/10/10 tree-disjoint split; length-eligible samples capped only within their assigned split",
        "selected_pair_identifiers_sha256": hashlib.sha256(json.dumps(identifiers).encode("utf-8")).hexdigest(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license_claim": "Apache-2.0 per official model card; re-check before redistribution/commercial use",
        "raw_text_written_to_artifact": False,
        "source_data_uploaded_to_artifact": False,
    })

    print("[4/7] Scoring base model on validation and acceptance gate", flush=True)
    base_val = score_responses(model, encoded["validation"])
    base_gate = score_responses(model, encoded["gate"])

    print("[5/7] Training LoRA adapter", flush=True)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    train_args = TrainingArguments(
        output_dir=str(output / "trainer_tmp"),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_steps=0,
        logging_strategy="steps",
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=max(1, min(4, args.max_steps // 2 or 1)),
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=train_args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
        data_collator=lambda batch: collate(batch, int(tokenizer.pad_token_id)), tokenizer=tokenizer,
    )
    train_result = trainer.train()
    write_json(output / "training_log.json", {
        "trainer_log_history": trainer.state.log_history,
        "train_metrics": train_result.metrics,
        "global_step": int(trainer.state.global_step),
        "max_steps": args.max_steps,
        "seed": args.seed,
        "train_and_validation_loss_available_by_step": True,
        "full_100_500_2000_examples_by_1_2_3_epochs_grid_completed": False,
    })

    print("[6/7] Applying paired tree-level acceptance gate", flush=True)
    cand_val = score_responses(model, encoded["validation"])
    cand_gate = score_responses(model, encoded["gate"])
    gate = paired_group_bootstrap(
        base_gate, cand_gate, seed=args.seed, samples=args.bootstrap_samples,
        margin=args.gate_margin, minimum_groups=args.minimum_gate_groups,
    )
    adapter_dir = output / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    print("[7/7] Gate fixed; opening final test for report-only metrics", flush=True)
    with model.disable_adapter():
        base_test = score_responses(model, encoded["test"])
    cand_test = score_responses(model, encoded["test"])
    base_val_nll, cand_val_nll = mean_nll(base_val), mean_nll(cand_val)
    base_test_nll, cand_test_nll = mean_nll(base_test), mean_nll(cand_test)
    report = {
        "project": "AMORPH isolated LLM SFT/LoRA proof-of-concept",
        "status": "completed",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "adapter": "LoRA r=8 alpha=16, q/k/v/o projections"},
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "language": "ru"},
        "objective": "Unweighted mean over conversation trees of per-example answer-token mean NLL; lower is better",
        "validation": {
            "base_mean_nll": base_val_nll,
            "adapter_mean_nll": cand_val_nll,
            "delta_adapter_minus_base": cand_val_nll - base_val_nll,
        },
        "quality_gate": gate,
        "final_test_report_only": {
            "base_mean_nll": base_test_nll,
            "adapter_mean_nll": cand_test_nll,
            "delta_adapter_minus_base": cand_test_nll - base_test_nll,
            "used_for_selection": False,
            "warning": "Final-test metrics are now observed; never tune on this same split.",
        },
        "training": {
            "seed": args.seed,
            "max_steps": args.max_steps,
            "actual_steps": int(trainer.state.global_step),
            "encoded_split_sizes": {name: len(group) for name, group in encoded.items()},
            "max_length": args.max_length,
            "elapsed_seconds": round(time.time() - started, 2),
            "only_lora_parameters_trained": True,
            "external_mera_ifeval_evaluation_run": False,
            "full_overfit_grid_run": False,
        },
        "limitations": [
            "Small bounded technical SFT run, not evidence of broad assistant quality.",
            "Does not access or fine-tune ChatGPT, Grok, or Gemini weights.",
            "NLL on OASST1 is not an external instruction-following, correctness, or safety metric.",
            "A small gate may be INCONCLUSIVE; that is not success.",
            "Loss logs cover only this capped run, not the full overfit grid or repeated-seed study.",
            "No monetary TCO optimization is claimed.",
            "Raw dialogue text and downloaded model cache are not copied to the artifact.",
            "Artifact access follows public repository and Actions permissions; do not put private data in this workflow.",
        ],
    }
    write_json(output / "benchmark_result.json", report)
    shutil.rmtree(output / "trainer_tmp", ignore_errors=True)
    write_inventory(output)
    print(json.dumps({
        "status": report["status"],
        "gate_verdict": gate["verdict"],
        "base_validation_nll": base_val_nll,
        "adapter_validation_nll": cand_val_nll,
        "gate_delta": gate["delta_mean"],
        "gate_ci": [gate["ci_lower"], gate["ci_upper"]],
        "output_dir": str(output),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
