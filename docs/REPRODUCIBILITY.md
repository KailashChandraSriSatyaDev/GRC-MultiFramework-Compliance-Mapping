# Reproducibility

## Environment

- Python 3.13.11 in a dedicated environment, packages installed only from
  `requirements/grc-mvp.lock.txt` (54 packages, hash-pinned; torch 2.10.0+cu130, numpy 2.4.1,
  scipy 1.16.3, pandas 2.3.3, scikit-learn 1.7.1, transformers 5.3.0, sentence-transformers 5.3.0,
  openpyxl 3.1.5, pytest 9.2.0) with
  `--require-hashes --no-deps --only-binary=:all: --index-url https://pypi.org/simple`.
- This project is **not installed** into the environment. Run it from the repository root:
  `PYTHONPATH=src python -m grc_mapper ...`; pytest finds `src/` through `pyproject.toml`.
- No code path downloads or installs anything at runtime.
- `tools/validate_env.py` checks the interpreter, the lock pins, the OpenMP/MKL inventory, CUDA
  detection (report only) and the offline `OK_FULL` model load.

## Local configuration

Copy `config/example.local.toml` outside the repository, fill in real paths and set `GRC_CONFIG`
to it. Resolution order: command-line options > environment variables > the TOML file >
built-in non-path defaults. Paths have no defaults. Outputs must be outside the repository.

## Model runtime manifest

A locally obtained copy of the production model must match these runtime files exactly
(`manifests/model_runtime_manifest.json`). The two expected non-runtime files (model card and
evaluation CSV) are verified against a local full manifest that is not published.

| File | Bytes | SHA-256 |
|---|---|---|
| `1_Pooling/config.json` | 321 | `b4105bf48c5257a542a7e1968cef220b3ecd49ae24943143279650c4ac6e30a0` |
| `config.json` | 774 | `cf566f0a2e73d887c44858fa7ebc28d87dfef1d52ebd48cc0e1c462ca7ccf465` |
| `config_sentence_transformers.json` | 294 | `55114feb391e1230474fcdfdf05ff629774e558da8ac793044a9063d092c4070` |
| `model.safetensors` | 90,864,176 | `c7c902984b4b5a837fd006dc8cce2270fa2d5287bb1d813a868f0cf8d5384894` |
| `modules.json` | 368 | `e7989e94b5b809d895a9521b708312c1ccd333e183effebaf3838908da2acd53` |
| `sentence_bert_config.json` | 60 | `5a78a8637a4c986a89b5e9b3ce4af84b94b1dfeb9f1dfb09c448182a9d7ca9ab` |
| `tokenizer.json` | 711,649 | `851ca67100d372ca3ae031a6abd168f53489eebfd7d89523f35c5c9b4d372c3c` |
| `tokenizer_config.json` | 588 | `6d5b6b07eeacce3dd24681a5575b404f9b92d0984f6d7e056dfc6b12924d177f` |

## Determinism

- Similarities are accumulated in float64 and stored as float32, so they do not depend on batch
  composition; ranking ties resolve by natural-sort control position.
- The fake demo encoder is seeded from SHA-256 of the text (never Python's `hash()`).
- Fine-tuning itself is not re-run here; the research training runs were unseeded (see LIMITATIONS.md).

## Comparing runs

A rerun reproduces a previous run only when the model weights, framework catalogue and version, text
handling, bands and weights are identical; `history.check_compatible` enforces these and rejects any
other pair. On the same statement text, the remaining dDI (DI_current - DI_previous) is numerical
noise: about 1e-7 between batch and single mapping, and about 5e-5 against historical files stored at
four decimals. Such a dDI is reproducibility evidence, not drift.
