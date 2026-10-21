import json

import numpy as np

from amorph_autotrainer.strategies import E5Candidate, load_strategy_model, predict_loaded


class FakeEncoder:
    """Deterministic encoder replacement so this test makes no network calls."""

    def __init__(self):
        self.seen_texts = []

    def encode(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        self.seen_texts.extend(texts)
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if "alpha" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "beta" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.5, 0.5, 0.1])
        return np.asarray(vectors, dtype=np.float32)


def test_e5_fit_save_and_load_round_trip_without_network(tmp_path, monkeypatch):
    encoder = FakeEncoder()
    monkeypatch.setattr(
        E5Candidate, "_load_encoder", staticmethod(lambda _model_id, _revision=None: encoder)
    )
    train_texts = [
        f"alpha training example {i}" if i % 2 == 0 else f"beta training example {i}"
        for i in range(24)
    ]
    train_labels = np.asarray(["alpha" if i % 2 == 0 else "beta" for i in range(24)])
    val_texts = ["alpha validation example", "beta validation example"]

    candidate = E5Candidate.fit(
        train_texts, train_labels, val_texts,
        model_id="fake/local-test", model_revision="fake-revision-sha",
    )
    assert len(candidate.val_pred) == 2
    assert candidate.latency_seconds_per_example >= 0
    assert encoder.seen_texts
    assert all(text.startswith("query: ") for text in encoder.seen_texts)

    model_dir = tmp_path / "model"
    metadata = E5Candidate.save(candidate, model_dir)
    (model_dir / "manifest.json").write_text(json.dumps({
        "strategy": "e5",
        "artifact_subdirectory": ".",
        "strategy_metadata": metadata,
    }), encoding="utf-8")

    loaded = load_strategy_model(model_dir)
    predictions = predict_loaded(loaded, val_texts)
    assert predictions.tolist() == ["alpha", "beta"]
    assert loaded["manifest"]["strategy_metadata"]["model_id"] == "fake/local-test"
    assert loaded["manifest"]["strategy_metadata"]["model_revision"] == "fake-revision-sha"
