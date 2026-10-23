"""Deterministic OASST1 Russian pair preparation with conversation-tree isolation."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable


def _stable_key(seed: int, *parts: str) -> str:
    payload = "\x1f".join([str(seed), *parts]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _flag_is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def prepare_ru_pairs(rows: Iterable[dict[str, Any]], *, seed: int = 42) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Select reviewed Russian prompter→assistant pairs, at most one per conversation tree."""
    materialized = [dict(row) for row in rows]
    by_id: dict[str, dict[str, Any]] = {}
    ambiguous_ids: set[str] = set()
    duplicate_ids = 0
    for row in materialized:
        message_id = str(row.get("message_id") or "")
        if not message_id or message_id in ambiguous_ids:
            continue
        previous = by_id.get(message_id)
        if previous is not None:
            fields = ("parent_id", "message_tree_id", "text", "role", "lang")
            if any(previous.get(k) != row.get(k) for k in fields):
                by_id.pop(message_id, None)
                ambiguous_ids.add(message_id)
                duplicate_ids += 1
            continue
        by_id[message_id] = row

    eligible: list[dict[str, str]] = []
    for reply in by_id.values():
        if str(reply.get("role") or "").lower() != "assistant":
            continue
        if str(reply.get("lang") or "").lower() != "ru":
            continue
        if _flag_is_true(reply.get("deleted")) or _flag_is_true(reply.get("synthetic")):
            continue
        if str(reply.get("tree_state") or "") != "ready_for_export":
            continue
        if reply.get("review_result") is not True and str(reply.get("review_result")).lower() != "true":
            continue
        answer = str(reply.get("text") or "").strip()
        tree_id = str(reply.get("message_tree_id") or "")
        parent_id = str(reply.get("parent_id") or "")
        parent = by_id.get(parent_id)
        if not answer or not tree_id or parent is None:
            continue
        if str(parent.get("role") or "").lower() != "prompter" or str(parent.get("lang") or "").lower() != "ru":
            continue
        if str(parent.get("message_tree_id") or "") != tree_id:
            continue
        if _flag_is_true(parent.get("deleted")) or str(parent.get("tree_state") or "") != "ready_for_export":
            continue
        prompt = str(parent.get("text") or "").strip()
        if not prompt:
            continue
        eligible.append({
            "tree_id": tree_id,
            "prompt_id": parent_id,
            "reply_id": str(reply.get("message_id")),
            "prompt": prompt,
            "response": answer,
        })

    chosen: dict[str, dict[str, str]] = {}
    for row in eligible:
        key = row["tree_id"]
        current = chosen.get(key)
        rank = _stable_key(seed, key, row["prompt_id"], row["reply_id"])
        if current is None or rank < _stable_key(seed, key, current["prompt_id"], current["reply_id"]):
            chosen[key] = row
    pairs = [chosen[key] for key in sorted(chosen, key=lambda tree: _stable_key(seed, tree))]
    return pairs, {
        "input_rows": len(materialized),
        "unique_message_ids": len(by_id),
        "conflicting_duplicate_message_ids": duplicate_ids,
        "eligible_direct_ru_pairs": len(eligible),
        "selected_pairs_one_per_tree": len(pairs),
        "selected_conversation_trees": len({p["tree_id"] for p in pairs}),
    }


def split_by_tree(pairs: list[dict[str, str]], *, seed: int = 42) -> dict[str, list[dict[str, str]]]:
    """Deterministic 70/10/10/10 split by tree; every tree appears in one split only."""
    tree_ids = [p["tree_id"] for p in pairs]
    if len(tree_ids) != len(set(tree_ids)):
        raise ValueError("Expected at most one pair per conversation tree")
    ordered = sorted(pairs, key=lambda p: _stable_key(seed, p["tree_id"]))
    n = len(ordered)
    n_train, n_val, n_gate = int(n * 0.70), int(n * 0.10), int(n * 0.10)
    return {
        "train": ordered[:n_train],
        "validation": ordered[n_train:n_train + n_val],
        "gate": ordered[n_train + n_val:n_train + n_val + n_gate],
        "test": ordered[n_train + n_val + n_gate:],
    }


def cap_split(pairs: list[dict[str, str]], *, limit: int, seed: int, split_name: str) -> list[dict[str, str]]:
    """Stable cap after tree-level split; no tree is moved between partitions."""
    if limit < 1:
        raise ValueError("split limit must be >= 1")
    return sorted(pairs, key=lambda p: _stable_key(seed, split_name, p["tree_id"]))[:limit]


def encode_pairs(pairs: list[dict[str, str]], tokenizer: Any, *, max_length: int) -> tuple[list[dict[str, Any]], int]:
    """Tokenize pairs and compute loss only over answer tokens."""
    encoded: list[dict[str, Any]] = []
    dropped = 0
    for pair in pairs:
        prompt_messages = [{"role": "user", "content": pair["prompt"]}]
        full_messages = prompt_messages + [{"role": "assistant", "content": pair["response"]}]
        prompt_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False)
        prompt_ids = list(prompt_ids)
        full_ids = list(full_ids)
        if len(full_ids) > max_length or len(full_ids) <= len(prompt_ids) + 1:
            dropped += 1
            continue
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Chat template changed the prompt prefix; refusing unsafe loss mask")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        encoded.append({
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "tree_id": pair["tree_id"],
        })
    return encoded, dropped
