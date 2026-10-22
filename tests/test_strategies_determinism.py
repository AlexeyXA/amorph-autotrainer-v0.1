import numpy as np

from amorph_autotrainer.strategies import DeterministicTfidfVectorizer


def test_max_features_ties_resolve_by_lexicographic_feature_name():
    texts = [
        "alpha beta",
        "alpha beta",
        "beta gamma",
        "gamma delta",
    ]

    first = DeterministicTfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        max_features=2,
        lowercase=False,
        use_idf=False,
        norm=None,
    )
    reversed_rows = DeterministicTfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        max_features=2,
        lowercase=False,
        use_idf=False,
        norm=None,
    )

    first.fit(texts)
    reversed_rows.fit(list(reversed(texts)))

    # "beta" has the highest frequency; "alpha" wins the tie with "gamma".
    assert first.vocabulary_ == {"alpha": 0, "beta": 1}
    assert reversed_rows.vocabulary_ == first.vocabulary_
    assert np.array_equal(first.get_feature_names_out(), np.array(["alpha", "beta"]))


def test_deterministic_vectorizer_keeps_top_frequency_before_tie_break():
    vectorizer = DeterministicTfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        max_features=2,
        lowercase=False,
        use_idf=False,
        norm=None,
    )
    vectorizer.fit(["zeta beta", "beta alpha", "beta gamma", "alpha"])

    assert set(vectorizer.vocabulary_) == {"alpha", "beta"}
