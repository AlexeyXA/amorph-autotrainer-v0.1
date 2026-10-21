# AMORPH AutoTrainer v0.1

Budget-aware, reproducible AutoTrainer MVP for **text classification and feature extraction**.

- Primary route: TF-IDF + Logistic Regression
- Optional route: `intfloat/multilingual-e5-small` embeddings + linear classifier
- Four-way split (train / validation / acceptance gate / final test)
- Quality gate with paired class-stratified bootstrap
- TCO estimates, SQLite experiment log, Colab notebook

**Benchmark protocol:** [BENCHMARK_PROTOCOL_RU.md](BENCHMARK_PROTOCOL_RU.md)

## Quick start

```bash
pip install -e ".[test]"
# optional E5:
# pip install -e ".[embeddings]"

amorph-autotrainer demo --output demo.csv --examples-per-class 50
amorph-autotrainer train demo.csv --text-column text --label-column label --out runs --strategies tfidf
```

## Package layout

- `amorph_autotrainer/` — core library (data, pipeline, strategies, metrics, cost, CLI)
- `tests/` — unit tests
- `scripts/benchmark_massive_ru.py` — MASSIVE ru-RU external benchmark (upload if missing)
- `notebooks/` — Colab notebook (upload if missing)

## License

MIT
