from experiments.llm_sft_oasst1.data import cap_split, prepare_ru_pairs, split_by_tree


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
        _row("u-synth", None, "prompter", "tree-synth", "Синтетический вопрос"),
        _row("a-synth", "u-synth", "assistant", "tree-synth", "Синтетический ответ", synthetic=True),
    ]
    pairs, audit = prepare_ru_pairs(rows, seed=9)
    assert len(pairs) == 100
    assert len({p["tree_id"] for p in pairs}) == len(pairs)
    assert audit["selected_pairs_one_per_tree"] == 100


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
    from experiments.llm_sft_oasst1.data import encode_pairs

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


def test_conflicting_duplicate_message_id_is_removed():
    rows = _rows(5)
    original = next(row for row in rows if row["message_id"] == "a-0000")
    rows.append({**original, "text": "Другая версия того же message ID"})
    pairs, audit = prepare_ru_pairs(rows)
    assert "tree-0000" not in {p["tree_id"] for p in pairs}
    assert audit["conflicting_duplicate_message_ids"] == 1
