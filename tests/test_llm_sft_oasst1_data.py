from experiments.llm_sft_oasst1.data import (
    cap_split, encode_pairs, encode_split_up_to_cap, prepare_ru_pairs, split_by_tree,
)


def _row(mid, parent, role, tree, text, *, lang="ru", review=True, deleted=False, synthetic=False):
    return {
        "message_id": mid, "parent_id": parent, "message_tree_id": tree,
        "role": role, "lang": lang, "text": text, "tree_state": "ready_for_export",
        "review_result": review, "deleted": deleted, "synthetic": synthetic,
    }


def _rows(n=100):
    result = []
    for i in range(n):
        tree = f"tree-{i:04}"
        user_id = f"u-{i:04}"
        result.append(_row(user_id, None, "prompter", tree, f"Вопрос {i}"))
        result.append(_row(f"a-{i:04}", user_id, "assistant", tree, f"Ответ {i}"))
    return result


def test_pairs_are_russian_reviewed_non_synthetic_and_one_per_tree():
    rows = _rows() + [
        _row("u-en", None, "prompter", "tree-en", "Hello", lang="en"),
        _row("a-en", "u-en", "assistant", "tree-en", "Hi", lang="en"),
        _row("u-bad", None, "prompter", "tree-bad", "Вопрос плохой"),
        _row("a-bad", "u-bad", "assistant", "tree-bad", "Ответ плохой", review=False),
        _row("u-synth-answer", None, "prompter", "tree-synth-answer", "Вопрос синтетического ответа"),
        _row("a-synth-answer", "u-synth-answer", "assistant", "tree-synth-answer", "Ответ синтетический", synthetic=True),
        _row("u-synth-parent", None, "prompter", "tree-synth-parent", "Синтетический prompt", synthetic=True),
        _row("a-synth-parent", "u-synth-parent", "assistant", "tree-synth-parent", "Ответ на синтетический prompt"),
        _row("u-unreviewed-parent", None, "prompter", "tree-unreviewed-parent", "Непроверенный prompt", review=False),
        _row("a-unreviewed-parent", "u-unreviewed-parent", "assistant", "tree-unreviewed-parent", "Ответ на непроверенный prompt"),
    ]
    pairs, audit = prepare_ru_pairs(rows, seed=9)
    assert len(pairs) == 100
    assert len({p["tree_id"] for p in pairs}) == len(pairs)
    assert audit["selected_pairs_one_per_tree_and_exact_prompt"] == 100
    assert audit["selected_conversation_trees_before_prompt_dedup"] == 100


def test_exact_normalized_prompts_are_deduplicated_across_trees():
    rows = _rows(10) + [
        _row("u-dup", None, "prompter", "tree-duplicate", "  ВОПРОС   3  "),
        _row("a-dup", "u-dup", "assistant", "tree-duplicate", "Другой ответ"),
    ]
    pairs, audit = prepare_ru_pairs(rows, seed=42)
    prompt_keys = [" ".join(p["prompt"].casefold().split()) for p in pairs]
    assert len(prompt_keys) == len(set(prompt_keys))
    assert audit["exact_normalized_prompt_duplicates_removed"] == 1


def test_splits_are_deterministic_and_tree_disjoint():
    pairs, _ = prepare_ru_pairs(_rows(100), seed=17)
    first, second = split_by_tree(pairs, seed=17), split_by_tree(pairs, seed=17)
    assert {k: [x["tree_id"] for x in v] for k, v in first.items()} == {
        k: [x["tree_id"] for x in v] for k, v in second.items()
    }
    groups = [{x["tree_id"] for x in first[k]} for k in ("train", "validation", "gate", "test")]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert groups[i].isdisjoint(groups[j])
    assert sum(len(v) for v in first.values()) == 100


def test_cap_is_stable():
    pairs, _ = prepare_ru_pairs(_rows(40))
    a = cap_split(pairs, limit=7, seed=42, split_name="train")
    b = cap_split(pairs, limit=7, seed=42, split_name="train")
    assert [x["tree_id"] for x in a] == [x["tree_id"] for x in b]
    assert len(a) == 7


def test_tokenization_masks_prompt_and_keeps_response_labels():
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            prompt = [101, len(messages[0]["content"])]
            if len(messages) == 1 and add_generation_prompt:
                return prompt
            return prompt + [202, len(messages[1]["content"]), 2]

    examples, dropped = encode_pairs(
        [{"tree_id": "one", "prompt": "Q?", "response": "A!"}],
        FakeTokenizer(), max_length=16,
    )
    assert dropped == 0
    assert examples[0]["labels"][:2] == [-100, -100]
    assert examples[0]["labels"][2:] == [202, 2, 2]


def test_split_encoder_scans_past_overlength_rows_without_cross_split_reassignment():
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            content = messages[0]["content"]
            if len(messages) == 1 and add_generation_prompt:
                return [101] + [7] * len(content)
            return [101] + [7] * len(content) + [202, 9, 2]

    pairs = [
        {"tree_id": "long-1", "prompt": "long prompt", "response": "ok"},
        {"tree_id": "short-1", "prompt": "q", "response": "a"},
        {"tree_id": "long-2", "prompt": "long prompt", "response": "ok"},
        {"tree_id": "short-2", "prompt": "q2", "response": "a"},
    ]
    encoded, stats = encode_split_up_to_cap(
        pairs, FakeTokenizer(), limit=2, max_length=5, seed=4, split_name="train"
    )
    assert len(encoded) == 2
    assert stats["candidate_trees_scanned"] >= 2
    assert stats["encoded_examples"] == 2
    assert len({r["tree_id"] for r in encoded}) == 2


def test_conflicting_duplicate_message_id_is_removed():
    rows = _rows(5)
    original = next(row for row in rows if row["message_id"] == "a-0000")
    rows.append({**original, "text": "Другая версия того же message ID"})
    pairs, audit = prepare_ru_pairs(rows)
    assert "tree-0000" not in {p["tree_id"] for p in pairs}
    assert audit["conflicting_duplicate_message_ids"] == 1
