from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression

from .strategy_abi import ComponentStatus, strategy_fingerprint


@dataclass
class FittedCandidate:
    name: str
    model: Any
    fit_seconds: float
    val_predict_seconds: float
    latency_seconds_per_example: float
    val_pred: np.ndarray
    metadata: dict[str, Any]


class TfidfCandidate:
    name = "tfidf"

    @classmethod
    def identity(cls) -> str:
        return cls.name

    @classmethod
    def version(cls) -> str:
        return "0.1.0"

    @classmethod
    def status(cls) -> ComponentStatus:
        return ComponentStatus.CANONICAL

    @classmethod
    def capabilities(cls) -> dict[str, Any]:
        return {
            "requires_gpu": False,
            "optional_extra": None,
            "feature_type": "word_char_tfidf",
            "supports_batch_prediction": True,
        }

    @classmethod
    def fingerprint(cls, *, code_revision: str | None = None) -> str:
        return strategy_fingerprint(
            identity=cls.identity(),
            version=cls.version(),
            hyperparameters={
                "word_ngram_range": [1, 2],
                "char_ngram_range": [3, 5],
                "max_features_per_vectorizer": 50000,
                "logistic_regression_C": 1.0,
                "max_iter": 1000,
                "class_weight": None,
            },
            code_revision=code_revision,
            extra=cls.capabilities(),
        )

    @staticmethod
    def build() -> Pipeline:
        features = FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), max_features=50_000,
                min_df=1, sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), max_features=50_000,
                min_df=1, sublinear_tf=True,
            )),
        ])
        return Pipeline([
            ("features", features),
            ("classifier", LogisticRegression(max_iter=1000, class_weight=None)),
        ])

    @classmethod
    def fit(cls, train_texts: list[str], train_labels: np.ndarray,
            val_texts: list[str]) -> FittedCandidate:
        model = cls.build()
        started = time.perf_counter()
        model.fit(train_texts, train_labels)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        val_pred = model.predict(val_texts)
        predict_seconds = time.perf_counter() - started
        probe = val_texts[:min(16, len(val_texts))]
        latency_started = time.perf_counter()
        for _ in range(3):
            for text in probe:
                model.predict([text])
        latency = (time.perf_counter() - latency_started) / (3 * max(1, len(probe)))
        return FittedCandidate(
            name=cls.name, model=model, fit_seconds=fit_seconds,
            val_predict_seconds=predict_seconds,
            latency_seconds_per_example=latency,
            val_pred=val_pred,
            metadata={"feature_type": "tfidf_word_and_char", "max_features_per_vectorizer": 50000},
        )

    @staticmethod
    def save(candidate: FittedCandidate, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(candidate.model, directory / "model.joblib")
        return {"strategy": "tfidf", "model_file": "model.joblib", **candidate.metadata}


class E5Candidate:
    """Optional embedding + linear classification strategy; may download model weights."""
    name = "e5"
    model_id = "intfloat/multilingual-e5-small"
    # Pinned Hugging Face commit observed on the official model repository on 2026-10-20.
    default_model_revision = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

    @classmethod
    def identity(cls) -> str:
        return cls.name

    @classmethod
    def version(cls) -> str:
        return "0.1.0"

    @classmethod
    def status(cls) -> ComponentStatus:
        return ComponentStatus.EXPERIMENTAL

    @classmethod
    def capabilities(cls) -> dict[str, Any]:
        return {
            "requires_gpu": False,
            "optional_extra": "embeddings",
            "feature_type": "multilingual_e5_embeddings",
            "supports_batch_prediction": True,
        }

    @classmethod
    def fingerprint(
        cls,
        *,
        code_revision: str | None = None,
        model_id: str | None = None,
        model_revision: str | None = None,
    ) -> str:
        return strategy_fingerprint(
            identity=cls.identity(),
            version=cls.version(),
            hyperparameters={
                "prefix": "query: ",
                "normalize_embeddings": True,
                "batch_size": 32,
                "classifier": "LogisticRegression",
                "classifier_C": 1.0,
                "classifier_max_iter": 1000,
            },
            code_revision=code_revision,
            model_revision=model_revision or cls.default_model_revision,
            extra={**cls.capabilities(), "model_id": model_id or cls.model_id},
        )

    @staticmethod
    def _load_encoder(model_id: str, model_revision: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "E5 strategy is optional. Install with `pip install '.[embeddings]'`."
            ) from exc
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(model_id, device=device, revision=model_revision)

    @staticmethod
    def _encode(encoder: Any, texts: list[str]) -> np.ndarray:
        # The multilingual-E5 model card recommends the `query:` prefix when
        # using embeddings as features (e.g. linear-probing classification).
        return encoder.encode(
            ["query: " + text for text in texts],
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    @classmethod
    def fit(cls, train_texts: list[str], train_labels: np.ndarray,
            val_texts: list[str], model_id: str | None = None,
            model_revision: str | None = None) -> FittedCandidate:
        model_id = model_id or cls.model_id
        model_revision = model_revision or cls.default_model_revision
        total_started = time.perf_counter()
        encoder = cls._load_encoder(model_id, model_revision)
        load_elapsed = time.perf_counter() - total_started

        started = time.perf_counter()
        x_train = cls._encode(encoder, train_texts)
        x_val = cls._encode(encoder, val_texts)
        feature_elapsed = time.perf_counter() - started

        classifier = LogisticRegression(max_iter=1000)
        started = time.perf_counter()
        classifier.fit(x_train, train_labels)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        val_pred = classifier.predict(x_val)
        predict_seconds = time.perf_counter() - started

        # Latency must include embedding generation plus classification, not only
        # the linear head. Use a small repeated per-example probe, like TF-IDF.
        probe = val_texts[:min(16, len(val_texts))]
        latency_started = time.perf_counter()
        for _ in range(3):
            for text in probe:
                features = cls._encode(encoder, [text])
                classifier.predict(features)
        latency = (time.perf_counter() - latency_started) / (3 * max(1, len(probe)))

        return FittedCandidate(
            name=cls.name, model={"classifier": classifier, "encoder": encoder,
                                  "model_id": model_id, "model_revision": model_revision},
            fit_seconds=fit_seconds + feature_elapsed + load_elapsed,
            val_predict_seconds=predict_seconds,
            latency_seconds_per_example=latency,
            val_pred=val_pred,
            metadata={"feature_type": "multilingual_e5_embeddings", "model_id": model_id,
                      "model_revision": model_revision,
                      "encoder_load_seconds": load_elapsed,
                      "feature_extraction_seconds": feature_elapsed,
                      "embedding_dim": int(x_train.shape[1])},
        )

    @staticmethod
    def save(candidate: FittedCandidate, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(candidate.model["classifier"], directory / "classifier.joblib")
        metadata = {
            "strategy": "e5",
            "model_file": "classifier.joblib",
            "model_id": candidate.model["model_id"],
            "model_revision": candidate.model.get("model_revision"),
            **candidate.metadata,
        }
        (directory / "strategy.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return metadata


def load_strategy_model(model_directory: str | Path):
    directory = Path(model_directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strategy = manifest["strategy"]
    artifact_dir = directory / manifest["artifact_subdirectory"]

    if strategy == "tfidf":
        return {"strategy": strategy, "model": joblib.load(artifact_dir / "model.joblib"),
                "manifest": manifest}
    if strategy == "e5":
        # New manifests nest strategy-specific metadata. The top-level fallback
        # keeps compatibility with any early bundles that used the old shape.
        strategy_metadata = manifest.get("strategy_metadata", {})
        model_id = manifest.get("model_id") or strategy_metadata.get("model_id")
        model_revision = strategy_metadata.get("model_revision")
        if not model_id:
            raise ValueError("E5 model manifest does not contain a model_id")
        encoder = E5Candidate._load_encoder(model_id, model_revision)
        classifier = joblib.load(artifact_dir / "classifier.joblib")
        return {"strategy": strategy, "model": classifier, "encoder": encoder,
                "manifest": manifest}
    raise ValueError(f"Unknown strategy in manifest: {strategy}")


def predict_loaded(loaded: dict[str, Any], texts: list[str]) -> np.ndarray:
    if loaded["strategy"] == "tfidf":
        return loaded["model"].predict(texts)
    features = E5Candidate._encode(loaded["encoder"], texts)
    return loaded["model"].predict(features)


def extract_loaded_features(loaded: dict[str, Any], texts: list[str]):
    if loaded["strategy"] == "tfidf":
        return loaded["model"].named_steps["features"].transform(texts)
    return E5Candidate._encode(loaded["encoder"], texts)
