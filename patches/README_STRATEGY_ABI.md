# Strategy ABI / conformance / replay (post-P0 hardening)

## On branch / local
- `amorph_autotrainer/strategy_abi.py` — ComponentStatus, StrategyABI Protocol, fingerprint, conformance, ReplayDescriptor
- `strategies.py` — TF-IDF CANONICAL, E5 EXPERIMENTAL; identity/version/capabilities/fingerprint
- `pipeline.py` — conformance before acceptance gate; writes `replay.json`; fingerprints + protocol_status in manifest
- `tests/test_strategy_abi.py`

## Apply
```bash
# after strategy_abi.py is on the tree:
patch -p1 < patches/0004-strategies-abi.patch   # partial if truncated; prefer full file copy
# pipeline: copy from worktree or apply pipeline_abi.patch when complete
pip install -e '.[test]'
pytest -q
```

Local verification (2026-10-22): **26 pytest passed** including e2e with `replay.json`.

Empty-input predict is soft for sklearn TF-IDF (unsupported); length/type checks are hard-fail.
