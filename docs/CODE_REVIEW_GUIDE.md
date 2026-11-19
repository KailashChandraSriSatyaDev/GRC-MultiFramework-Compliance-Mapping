# Code review guide

## What can be run without private assets

From the repository root, with Python 3.13 and the locked packages (numpy and pytest suffice for
the unit tests and the demo):

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -m unit
PYTHONPATH=src python -m grc_mapper demo
python tools/check_publishable.py            # INCOMPLETE without the local configuration (by design)
```

The demo uses `FakeEncoder` and `examples/synthetic/`. It is a software-pipeline demonstration:
its scores are not NLP inference and not research results.

## What needs the private local assets

Integration and regression tests and every real mapping command need the verified local model,
the control catalogue and a local configuration (`GRC_CONFIG`):

```text
python -m grc_mapper verify-model --load
python -m grc_mapper verify-inputs --profile research
python -m grc_mapper map-one --text-file statement.txt [--out <DIR>]
python -m grc_mapper map-batch --statements statements.csv --out <DIR>
python -m grc_mapper map-multi --statements statements.csv --frameworks <SPEC.json> --out <DIR> [--select-column COL --select-value VALUE]
python -m pytest -m integration
python -m pytest -m regression
python tools/check_publishable.py --final
```

On Windows PowerShell 5.1, prefer `--text-file` for real text. If you pipe into `--stdin`, first set
`$OutputEncoding = [Text.UTF8Encoding]::new($false)`. Run tests through `Start-Process` with stdout
and stderr redirected separately so that warnings on stderr do not become `NativeCommandError`s.

## Module map

| Module | Responsibility |
|---|---|
| `errors.py` | error types and CLI exit codes |
| `config.py` | research constants, settings resolution, strict TOML validation |
| `offline.py` | offline environment variables, application-level socket guard |
| `integrity.py` | manifests, strict model-directory verification, post-load re-verification |
| `data.py` | CSV loading and validation; natural-sort control catalogue |
| `text.py` | text validation (returns the text unchanged), `TokenProfile` |
| `model.py` | `OK_FULL`-gated offline loader, tokenizer-based token profiles |
| `mapping.py` | embedding checks, similarity, deterministic top-k |
| `scoring.py` | bands, DI_top1 (`drift_index`, 1 - top-1 similarity), APS = DI x tier weight, legacy rounding, priority queue |
| `pipeline.py` | the single mapping path used by batch and single-statement mapping |
| `textio.py` | `--text-file` / `--stdin` readers |
| `fake_encoder.py` | hash-seeded demo encoder |
| `export.py`, `provenance.py`, `logging_setup.py` | outputs, run records, redacting stderr logging |
| `csvcell.py` | CSV formula-injection guard (reversible `'` prefix; plain numbers unchanged) used by every CSV writer |
| `frameworks.py` | framework spec files (ID, name, version, catalogue, source, licence, bands, weights) and validated per-framework catalogues |
| `multiframework.py` | one encoding pass, then an independent index, ranking, bands, DI, APS and queue per framework |
| `history.py` | run snapshots keyed by statement ID and text hash; compatibility checks; previous-run comparison (dDI = DI_current - DI_previous) and change review queue |
| `cli.py`, `__main__.py` | `verify-model`, `verify-inputs`, `map-batch`, `map-one`, `demo`, `map-multi` |

## Exit codes

0 ok; 1 unexpected; 2 configuration or usage; 3 integrity; 4 input; 5 model; 6 write failure;
7 verification incomplete (`verify-model` only).

Production-model commands (`map-one`, `map-batch`, `map-multi`) exit **2** when no full model manifest
is configured (they stop before verification or any model-library import) and **3** when a full
manifest is configured but cannot be read or does not match. `OK_PARTIAL` therefore never permits
inference; `verify-model` reports it with exit 7.

## Suggested review focus

1. `integrity.verify_model_dir` and `model.load_verified_model`: order of checks, closed layout, `OK_FULL` gate.
2. `mapping.top_k` and `scoring.priority_queue`: determinism and tie handling.
3. `scoring.score_row`: band boundaries and the separation of unrounded and `_r4` values.
4. `pipeline.map_single`: identical path to `map_statements`.
5. `tools/check_publishable.py`: filesystem walk independent of Git, fail-closed status logic.
6. `history.check_compatible` and `compare_snapshots`: which fields block a comparison (framework ID and
   version, catalogue hash, model-weights hash, preprocessing, bands, weights) and which are only
   flagged (text hash per row).

Definitions of DI_top1, DI_mapped (not implemented), APS and dDI are in
[METHODOLOGY.md](METHODOLOGY.md), sections 5 and 7. `drift_index` is a legacy name for DI_top1.
