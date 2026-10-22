from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression

from .strategy_abi import ComponentStatus, strategy_fingerprint


class DeterministicTfidfVectorizer(TfidfVectorizer):
    """TfidfVectorizer with an explicit tie-break when max_features truncates vocabulary.

    sklearn ranks capped features by total term frequency, but equal-frequency
    features are otherwise selected through an unstable argsort tie. Resolve
    those ties by feature name so repeated runs retain the same vocabulary.
    """

    def _limit_features(self, X, vocabulary, high=None, low=None, limit=None):
        if high is None and low is None and limit is None:
            return X, set()

        # CountVectorizer stores one non-zero entry per feature and document.
        document_frequency = np.bincount(X.indices, minlength=X.shape[1])
        mask = np.ones(len(document_frequency), dtype=bool)
        if high is not None:
            mask &= document_frequency <= high
        if low is not None:
            mask &= document_frequency >= low

        if limit is not None and int(mask.sum()) > limit:
            term_frequency = np.asarray(X.sum(axis=0)).ravel()
            feature_name_by_index = {
                int(index): name for name, index in vocabulary.items()
            }
            eligible = np.flatnonzero(mask)
            chosen = sorted(
                eligible.tolist(),
                key=lambda index: (
                    -int(term_frequency[index]),
                    feature_name_by_index[index],
                ),
            )[:limit]
            stable_mask = np.zeros(len(document_frequency), dtype=bool)
            stable_mask[chosen] = True
            mask = stable_mask

        new_indices = np.cumsum(mask) - 1
        for term, old_index in list(vocabulary.items()):
            old_index = int(old_index)
            if mask[old_index]:
                vocabulary[term] = int(new_indices[old_index])
            else:
                del vocabulary[term]

        kept_indices = np.flatnonzero(mask)
        if kept_indices.size == 0:
            raise ValueError(
                "After pruning, no terms remain. Try a lower min_df or a higher max_df."
            )
        return X[:, kept_indices]


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
    def fingerprint(
        cls, *, code_revision: str | None = None,
        source_commit_sha: str | None = None,
    ) -> str:
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
                "max_features_tie_break": "term_frequency_desc_then_feature_name_asc",
            },
            code_revision=code_revision,
            source_commit_sha=source_commit_sha,
            extra=cls.capabilities(),
        )

    @staticmethod
    def build() -> Pipeline:
        features = FeatureUnion([
            ("word", DeterministicTfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), max_features=50_000,
                min_df=1, sublinear_tf=True,
            )),
            ("char", DeterministicTfidfVectorizer(
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
        classifier_c: float = 1.0,
        class_weight: str | None = None,
        source_commit_sha: str | None = None,
    ) -> str:
        return strategy_fingerprint(
            identity=cls.identity(),
            version=cls.version(),
            hyperparameters={
                "prefix": "query: ",
                "normalize_embeddings": True,
                "batch_size": 32,
                "classifier": "LogisticRegression",
                "classifier_C": classifier_c,
                "classifier_class_weight": class_weight,
                "classifier_max_iter": 1000,
            },
            code_revision=code_revision,
            source_commit_sha=source_commit_sha,
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
            model_revision: str | None = None, classifier_c: float = 1.0,
            class_weight: str | None = None) -> FittedCandidate:
        model_id = model_id or cls.model_id
        model_revision = model_revision or cls.default_model_revision
        if not math.isfinite(float(classifier_c)) or float(classifier_c) <= 0:
            raise ValueError("E5 classifier C must be a finite number > 0")
        if class_weight not in (None, "balanced"):
            raise ValueError("E5 class_weight must be None or 'balanced'")
        total_started = time.perf_counter()
        encoder = cls._load_encoder(model_id, model_revision)
        load_elapsed = time.perf_counter() - total_started

        started = time.perf_counter()
        x_train = cls._encode(encoder, train_texts)
        x_val = cls._encode(encoder, val_texts)
        feature_elapsed = time.perf_counter() - started

        classifier = LogisticRegression(
            C=float(classifier_c), class_weight=class_weight, max_iter=1000
        )
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
                                  "model_id": model_id, "model_revision": model_revision,
                                  "classifier_c": float(classifier_c), "class_weight": class_weight},
            fit_seconds=fit_seconds + feature_elapsed + load_elapsed,
            val_predict_seconds=predict_seconds,
            latency_seconds_per_example=latency,
            val_pred=val_pred,
            metadata={"feature_type": "multilingual_e5_embeddings", "model_id": model_id,
                      "model_revision": model_revision,
                      "classifier_C": float(classifier_c), "classifier_class_weight": class_weight,
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


class HybridTfidfE5Candidate:
    """Experimental sparse TF-IDF + normalized E5 feature concatenation strategy."""

    name = "hybrid_tfidf_e5"
    tfidf_global_scale = 1.0 / math.sqrt(2.0)

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
            "feature_type": "tfidf_word_char_plus_e5",
            "fusion": "sparse_horizontal_concatenation",
            "supports_batch_prediction": True,
        }

    @classmethod
    def fingerprint(
        cls,
        *,
        code_revision: str | None = None,
        source_commit_sha: str | None = None,
        model_id: str | None = None,
        model_revision: str | None = None,
        classifier_c: float = 1.0,
        class_weight: str | None = None,
    ) -> str:
        model_id = model_id or E5Candidate.model_id
        model_revision = model_revision or E5Candidate.default_model_revision
        return strategy_fingerprint(
            identity=cls.identity(),
            version=cls.version(),
            hyperparameters={
                "fusion": "sparse_hstack_tfidf_word_char_and_e5",
                "tfidf_word_ngram_range": [1, 2],
                "tfidf_char_ngram_range": [3, 5],
                "tfidf_max_features_per_vectorizer": 50000,
                "tfidf_max_features_tie_break": "term_frequency_desc_then_feature_name_asc",
                "tfidf_global_scale": cls.tfidf_global_scale,
                "e5_prefix": "query: ",
                "e5_normalize_embeddings": True,
                "e5_batch_size": 32,
                "classifier": "LogisticRegression",
                "classifier_C": classifier_c,
                "classifier_class_weight": class_weight,
                "classifier_max_iter": 1000,
            },
            code_revision=code_revision,
            source_commit_sha=source_commit_sha,
            model_revision=model_revision,
            extra={**cls.capabilities(), "model_id": model_id},
        )

    @classmethod
    def _join_features(cls, tfidf_features: Any, e5_embeddings: np.ndarray):
        dense = np.asarray(e5_embeddings, dtype=np.float32)
        if dense.ndim != 2:
            raise ValueError(f"E5 embeddings must be two-dimensional; got shape={dense.shape}")
        if tfidf_features.shape[0] != dense.shape[0]:
            raise ValueError(
                f"TF-IDF/E5 row mismatch: {tfidf_features.shape[0]} != {dense.shape[0]}"
            )
        if not np.isfinite(dense).all():
            raise ValueError("E5 embeddings contain non-finite values")
        # Word and char TF-IDF channels are individually L2-normalized. Scaling
        # their concatenation by 1/sqrt(2) makes the combined lexical block
        # approximately unit-norm, matching normalized E5 embeddings.
        lexical = tfidf_features * cls.tfidf_global_scale
        return sparse.hstack(
            (lexical, sparse.csr_matrix(dense)),
            format="csr",
        )

    @classmethod
    def fit(
        cls,
        train_texts: list[str],
        train_labels: np.ndarray,
        val_texts: list[str],
        model_id: str | None = None,
        model_revision: str | None = None,
        classifier_c: float = 1.0,
        class_weight: str | None = None,
    ) -> FittedCandidate:
        model_id = model_id or E5Candidate.model_id
        model_revision = model_revision or E5Candidate.default_model_revision
        if not math.isfinite(float(classifier_c)) or float(classifier_c) <= 0:
            raise ValueError("Hybrid classifier C must be a finite number > 0")
        if class_weight not in (None, "balanced"):
            raise ValueError("Hybrid class_weight must be None or 'balanced'")

        started_total = time.perf_counter()
        encoder = E5Candidate._load_encoder(model_id, model_revision)
        encoder_load_seconds = time.perf_counter() - started_total

        feature_started = time.perf_counter()
        tfidf_features = TfidfCandidate.build().named_steps["features"]
        x_train_tfidf = tfidf_features.fit_transform(train_texts)
        x_val_tfidf = tfidf_features.transform(val_texts)
        x_train_e5 = E5Candidate._encode(encoder, train_texts)
        x_val_e5 = E5Candidate._encode(encoder, val_texts)
        x_train = cls._join_features(x_train_tfidf, x_train_e5)
        x_val = cls._join_features(x_val_tfidf, x_val_e5)
        feature_extraction_seconds = time.perf_counter() - feature_started

        classifier = LogisticRegression(
            C=float(classifier_c),
            class_weight=class_weight,
            max_iter=1000,
        )
        fit_started = time.perf_counter()
        classifier.fit(x_train, train_labels)
        classifier_fit_seconds = time.perf_counter() - fit_started

        predict_started = time.perf_counter()
        val_pred = classifier.predict(x_val)
        predict_seconds = time.perf_counter() - predict_started

        probe = val_texts[:min(16, len(val_texts))]
        latency_started = time.perf_counter()
        for _ in range(3):
            for text in probe:
                lexical = tfidf_features.transform([text])
                dense = E5Candidate._encode(encoder, [text])
                classifier.predict(cls._join_features(lexical, dense))
        latency = (time.perf_counter() - latency_started) / (3 * max(1, len(probe)))

        model = {
            "classifier": classifier,
            "tfidf_features": tfidf_features,
            "encoder": encoder,
            "model_id": model_id,
            "model_revision": model_revision,
            "classifier_c": float(classifier_c),
            "class_weight": class_weight,
        }
        metadata = {
            "feature_type": "tfidf_word_char_plus_multilingual_e5",
            "fusion": "sparse_horizontal_concatenation",
            "tfidf_global_scale": cls.tfidf_global_scale,
            "model_id": model_id,
            "model_revision": model_revision,
            "classifier_C": float(classifier_c),
            "classifier_class_weight": class_weight,
            "encoder_load_seconds": encoder_load_seconds,
            "feature_extraction_seconds": feature_extraction_seconds,
            "classifier_fit_seconds": classifier_fit_seconds,
            "tfidf_feature_count": int(x_train_tfidf.shape[1]),
            "embedding_dim": int(x_train_e5.shape[1]),
            "combined_feature_count": int(x_train.shape[1]),
        }
        return FittedCandidate(
            name=cls.name,
            model=model,
            fit_seconds=encoder_load_seconds + feature_extraction_seconds + classifier_fit_seconds,
            val_predict_seconds=predict_seconds,
            latency_seconds_per_example=latency,
            val_pred=val_pred,
            metadata=metadata,
        )

    @classmethod
    def transform_features(cls, model: dict[str, Any], texts: list[str]):
        lexical = model["tfidf_features"].transform(texts)
        dense = E5Candidate._encode(model["encoder"], texts)
        return cls._join_features(lexical, dense)

    @classmethod
    def predict(cls, model: dict[str, Any], texts: list[str]) -> np.ndarray:
        features = cls.transform_features(model, texts)
        return model["classifier"].predict(features)

    @staticmethod
    def save(candidate: FittedCandidate, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(candidate.model["tfidf_features"], directory / "tfidf_features.joblib")
        joblib.dump(candidate.model["classifier"], directory / "classifier.joblib")
        metadata = {
            "strategy": "hybrid_tfidf_e5",
            "tfidf_features_file": "tfidf_features.joblib",
            "model_file": "classifier.joblib",
            "model_id": candidate.model["model_id"],
            "model_revision": candidate.model["model_revision"],
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
    if strategy == "hybrid_tfidf_e5":
        strategy_metadata = manifest.get("strategy_metadata", {})
        model_id = manifest.get("model_id") or strategy_metadata.get("model_id")
        model_revision = strategy_metadata.get("model_revision")
        if not model_id:
            raise ValueError("Hybrid TF-IDF/E5 manifest does not contain a model_id")
        encoder = E5Candidate._load_encoder(model_id, model_revision)
        model = {
            "classifier": joblib.load(artifact_dir / "classifier.joblib"),
            "tfidf_features": joblib.load(artifact_dir / "tfidf_features.joblib"),
            "encoder": encoder,
            "model_id": model_id,
            "model_revision": model_revision,
            "classifier_c": strategy_metadata.get("classifier_C", 1.0),
            "class_weight": strategy_metadata.get("classifier_class_weight"),
        }
        return {"strategy": strategy, "model": model, "manifest": manifest}
    raise ValueError(f"Unknown strategy in manifest: {strategy}")


def predict_loaded(loaded: dict[str, Any], texts: list[str]) -> np.ndarray:
    if loaded["strategy"] == "tfidf":
        return loaded["model"].predict(texts)
    if loaded["strategy"] == "hybrid_tfidf_e5":
        return HybridTfidfE5Candidate.predict(loaded["model"], texts)
    features = E5Candidate._encode(loaded["encoder"], texts)
    return loaded["model"].predict(features)


def extract_loaded_features(loaded: dict[str, Any], texts: list[str]):
    if loaded["strategy"] == "tfidf":
        return loaded["model"].named_steps["features"].transform(texts)
    if loaded["strategy"] == "hybrid_tfidf_e5":
        return HybridTfidfE5Candidate.transform_features(loaded["model"], texts)
    return E5Candidate._encode(loaded["encoder"], texts)
