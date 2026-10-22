import json

import numpy as np
from scipy import sparse

from amorph_autotrainer.cost import CostConfig
from amorph_autotrainer.demo import create_demo_csv
from amorph_autotrainer.pipeline import run_training, _validate_strategy_names
from amorph_autotrainer.strategies import (
    E5Candidate,
    HybridTfidfE5Candidate,
    extract_loaded_features,
    load_strategy_model,
    predict_loaded,
)
from amorph_autotrainer.strategy_abi import ComponentStatus


class FakeEncoder:
    """Deterministic local stand-in; no Hugging Face access or GPU is required."""

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
                vectors.append([1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)])
        return np.asarray(vectors, dtype=np.float32)


def _training_rows():
    texts = [
        f"alpha category training message token {i % 7} unique {i}"
        if i % 2 == 0
        else f"beta category training message token {i % 7} unique {i}"
        for i in range(40)
    ]
    labels = np.asarray(["alpha" if i % 2 == 0 else "beta" for i in range(40)])
    return texts, labels


def test_hybrid_strategy_fit_save_load_predict_and_feature_export(tmp_path, monkeypatch):
    encoder = FakeEncoder()
    monkeypatch.setattr(
        E5Candidate,
        "_load_encoder",
        staticmethod(lambda _model_id, _revision=None: encoder),
    )
    train_texts, labels = _training_rows()
    val_texts = ["alpha validation message", "beta validation message"]

    candidate = HybridTfidfE5Candidate.fit(
        train_texts,
        labels,
        val_texts,
        model_id="fake/local-test",
        model_revision="fake-revision-sha",
    )
    assert candidate.name == "hybrid_tfidf_e5"
    assert candidate.val_pred.tolist() == ["alpha", "beta"]
    assert candidate.metadata["fusion"] == "sparse_horizontal_concatenation"
    assert candidate.metadata["tfidf_global_scale"] == HybridTfidfE5Candidate.tfidf_global_scale
    assert candidate.metadata["combined_feature_count"] == (
        candidate.metadata["tfidf_feature_count"] + candidate.metadata["embedding_dim"]
    )
    assert encoder.seen_texts
    assert all(text.startswith("query: ") for text in encoder.seen_texts)

    model_dir = tmp_path / "hybrid-model"
    metadata = HybridTfidfE5Candidate.save(candidate, model_dir)
    assert (model_dir / "classifier.joblib").is_file()
    assert (model_dir / "tfidf_features.joblib").is_file()
    (model_dir / "manifest.json").write_text(
        json.dumps({
            "strategy": "hybrid_tfidf_e5",
            "artifact_subdirectory": ".",
            "strategy_metadata": metadata,
        }),
        encoding="utf-8",
    )

    loaded = load_strategy_model(model_dir)
    assert predict_loaded(loaded, val_texts).tolist() == ["alpha", "beta"]
    features = extract_loaded_features(loaded, val_texts)
    assert sparse.issparse(features)
    assert features.shape[0] == len(val_texts)
    assert features.shape[1] == metadata["combined_feature_count"]
    assert loaded["manifest"]["strategy_metadata"]["model_revision"] == "fake-revision-sha"


def test_hybrid_has_experimental_abi_and_fingerprint_includes_fusion_parameters():
    assert HybridTfidfE5Candidate.identity() == "hybrid_tfidf_e5"
    assert HybridTfidfE5Candidate.status() is ComponentStatus.EXPERIMENTAL
    baseline = HybridTfidfE5Candidate.fingerprint(
        code_revision="tree",
        source_commit_sha="commit",
        model_id="fake/model",
        model_revision="fake-revision",
        classifier_c=1.0,
        class_weight=None,
    )
    assert baseline == HybridTfidfE5Candidate.fingerprint(
        code_revision="tree",
        source_commit_sha="commit",
        model_id="fake/model",
        model_revision="fake-revision",
        classifier_c=1.0,
        class_weight=None,
    )
    assert baseline != HybridTfidfE5Candidate.fingerprint(
        code_revision="tree",
        source_commit_sha="commit",
        model_id="fake/model",
        model_revision="fake-revision",
        classifier_c=0.5,
        class_weight=None,
    )
    assert _validate_strategy_names(["hybrid_tfidf_e5"]) == ["tfidf", "hybrid_tfidf_e5"]


def test_pipeline_trains_hybrid_without_external_encoder_or_default_change(tmp_path, monkeypatch):
    encoder = FakeEncoder()
    monkeypatch.setattr(
        E5Candidate,
        "_load_encoder",
        staticmethod(lambda _model_id, _revision=None: encoder),
    )
    data_path = create_demo_csv(tmp_path / "demo.csv", examples_per_class=32)
    result = run_training(
        csv_path=data_path,
        output_dir=tmp_path / "runs",
        strategies=["hybrid_tfidf_e5"],
        config=CostConfig(
            max_runtime_seconds=300,
            max_candidates=2,
            expected_queries=10,
            bootstrap_samples=20,
            noninferiority_margin=0.02,
            min_gate_examples_per_class=3,
        ),
        max_near_duplicate_audit_rows=1,
    )
    run_dir = tmp_path / "runs" / result["run_id"]
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert "hybrid_tfidf_e5" in report["validation_metrics_all_candidates"]
    assert report["test_split_used_for_selection"] is False
    assert manifest["strategy_status"]["hybrid_tfidf_e5"] == "experimental"
    assert manifest["conformance_gate"]["hybrid_tfidf_e5"]["passed"] is True
    assert manifest["strategy_fingerprints"]["hybrid_tfidf_e5"]
    assert manifest["config"]["strategies"] == ["tfidf", "hybrid_tfidf_e5"]
    assert manifest["config"]["e5_classifier_c"] == 1.0
    assert manifest["config"]["e5_class_weight"] is None
