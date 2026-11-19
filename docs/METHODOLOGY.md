# Methodology

This document describes what the code computes. Evidence figures are in [RESULTS.md](RESULTS.md);
limitations are in [LIMITATIONS.md](LIMITATIONS.md).

## 1. Model and inputs

- **Encoder:** `all-MiniLM-L6-v2` fine-tuned with Multiple Negatives Ranking Loss on the 206
  research policies (the "production" model). Mean pooling, L2 normalisation, 384 dimensions,
  `max_seq_length` 256. The model files are not distributed (see [DATA_AND_LICENSING.md](DATA_AND_LICENSING.md)).
- **Control side:** the stored `Control_Text` of each of the 93 ISO/IEC 27001:2022 Annex A controls,
  passed to the encoder verbatim (identifier, title, purpose sentence, theme). No other field and
  no preprocessing are used.
- **Statement side:** the statement text, verbatim. Validation never alters the text.

## 2. Verification before inference

Every production-model run first verifies the model directory against two manifests:
the public runtime manifest (8 runtime files: path, size, SHA-256) and a local full manifest
(those 8 plus 2 expected non-runtime files). The directory layout is a closed list; any other file,
directory, symlink, junction or reparse point fails verification. Only `OK_FULL` permits loading.
`OK_PARTIAL` (no full manifest) is diagnostic only. Loading is offline (`HF_HUB_OFFLINE`,
`TRANSFORMERS_OFFLINE`, `local_files_only=True`, `trust_remote_code=False`) inside an
application-level socket guard. After loading, every file is re-hashed against the full manifest.

The socket guard patches Python's `socket` connect functions in this process only. It is not
OS-level network isolation.

## 3. Mapping

1. Controls and statements are encoded; embeddings are checked for shape, finiteness and unit norm.
2. Similarity is the dot product of the unit-norm float32 embeddings (equal to cosine similarity),
   accumulated in float64 and stored as float32. This makes a statement's similarities independent
   of how many statements are mapped together.
3. Candidates are ranked by similarity (descending), then by catalogue position. The catalogue is in
   natural-sort order of control ID (A.5.2 before A.5.10), so exact ties go to the control that sorts
   first and CSV row order never changes a result. Nothing is rounded before ranking.
4. The top-k candidates (default 3) are reported with their similarities.

Batch mapping and single-statement mapping are one implementation: `map_single` calls
`map_statements` with one statement.

## 4. Heuristic similarity review bands

The top-1 similarity (unrounded) is placed in a band:

| Code | Condition | Display label |
|---|---|---|
| `HIGH` | sim >= 0.59 | High similarity - candidate mapping |
| `REVIEW` | 0.49 <= sim < 0.59 | Moderate similarity - human review |
| `ESCALATE` | sim < 0.49 | Low similarity - escalate |

The cut-offs come from the out-of-fold fine-tuning analysis: an F1-optimal threshold of 0.50 plus
fixed offsets (+0.09 / -0.01), chosen on the same 206 out-of-fold scores and not nested. Cosine
similarity is not a probability and the bands are not probability-calibrated. They are triage bands
for review effort, not confidence levels. For other frameworks the bands come from the framework spec
(section 7).

## 5. Snapshot dissimilarity (DI) and Audit Priority Score (APS)

Terms used in every document of this repository:

| Term | Definition | Computed here? |
|---|---|---|
| **DI_top1** (field `drift_index`, function `scoring.drift_index`) | 1 - cosine similarity between a statement and the control the model ranks first, in one run | yes; this is the only DI the code computes |
| **DI_mapped** | 1 - cosine similarity between a statement and an independently assigned control (for example from a Statement of Applicability or an approved policy-control register) | **no**; no input carries an assigned control, so DI_mapped is unavailable. Neither the predicted control nor evaluation labels (`True_Control_ID`) may stand in for it |
| **APS** (field `aps`) | DI_top1 x tier weight | yes |
| **dDI** (`delta_di`, `history.py`) | DI_top1 in the current run - DI_top1 in a previous compatible run, per statement ID | yes, only between compatible snapshots (section 7) |

"DI" without a suffix means DI_top1. The field and function names `drift_index` are legacy names kept
for compatibility with the research notebooks and existing exports; the value is a **top-1 snapshot
dissimilarity index**. It is not temporal policy drift, not evidence that a policy changed, not evidence
of control failure or non-compliance, and not a calibrated probability of risk.

- **DI_top1 = 1 - top-1 similarity** (range [0, 2] in principle; cosine similarity can be negative).
  Because it is computed against the model's own first choice, a low DI says the statement is close to
  *some* control, not that it is close to the control an auditor assigned.
- **APS = DI_top1 x tier weight**, with configured weights HIGH 1.00, REVIEW 1.25, ESCALATE 1.50. APS
  is a heuristic audit-review priority score. Its only input is the top-1 similarity (through DI and the
  band); it does not combine independent risk signals, does not measure control criticality, and is
  neither a validated risk score nor a calibrated probability.
- **Ordering.** When the weights satisfy HIGH <= REVIEW <= ESCALATE (true for the defaults), APS is
  decreasing in top-1 similarity (non-increasing after floating-point rounding, with ties broken by
  ascending similarity), so the priority queue is exactly the statements in ascending order of top-1
  similarity. The configuration validator accepts any positive weights; with a
  non-monotonic setting (for example HIGH > REVIEW) statements just above a band edge can outrank
  statements just below it, and the equivalence no longer holds. APS values (not their order) jump at
  band edges, so APS magnitudes are not comparable to DI magnitudes.
- **Priority queue:** descending unrounded APS, then ascending top-1 similarity, then statement ID.

### Rounding

DI and APS are kept unrounded (float64) for ranking and research use. Separate legacy-compatible
values reproduce the research notebook exactly:
`drift_index_r4 = numpy.round(1 - sim, 4)` and `aps_r4 = numpy.round(drift_index_r4 x weight, 4)`
(sequential rounding with NumPy semantics; Python's built-in `round()` differs on half-way cases).
Queue ranks never depend on the rounded values.

## 6. Token profiles

For every control and every statement the real tokenizer reports the token count with special
tokens and without truncation (`n_tokens_full`) and the length the model actually receives
(`n_tokens_used`, at most 256). `truncated` is true when the two differ. Truncation is reported,
never hidden, and the input text is never shortened by this code.

## 7. Multiple frameworks

`map-multi` maps the same statements against several framework catalogues listed in a local spec file
(framework ID, name, version, catalogue file, source, licence, bands with their basis, and weights).
Statements are encoded once; each framework has its own control index, top-k ranking, review bands,
DI_top1, APS and queue, and every exported row carries `framework_id` and `framework_version`.
Framework identifiers are never merged and scores from different frameworks are never ranked together.

The model was fine-tuned on ISO/IEC 27001 control texts only, and the 0.59 / 0.49 cut-offs were derived
for ISO. A spec may reuse them for another framework (the local spec does so for NIST SP 800-53 Rev.
5.2.0, with the same weights); they are then **provisional triage settings, not calibrated for that
framework**, and the spec's `bands.basis` and the provenance say so.

Similarities from different catalogues do not have the same statistical meaning. The top-1 score is a
maximum over the catalogue: a larger catalogue (1,014 NIST controls against 93 ISO controls) raises the
expected maximum and narrows the top-1/top-2 margin, and closely related entries (a base control and
its enhancements) can sit within a few thousandths of each other. Band counts, mean similarity and DI
across frameworks therefore describe different retrieval problems and are not a measure of which
framework fits a statement better. A cross-framework comparison describes candidate-mapping
differences; it is not an equivalence mapping, a crosswalk or evidence of compliance coverage. An
external reference mapping (such as NIST's 2023 informative SP 800-53 to ISO/IEC 27001:2022 mapping)
can show whether two candidates appear as a listed pair; a listed pair does not establish equivalence
and an unlisted pair does not establish absence of a relation.

### Previous-run comparison (`history.py`)

A snapshot holds, per statement ID, the text SHA-256, top-1 control, top-1 similarity, band, DI_top1,
APS and the ranked top-k IDs when available. `check_compatible` requires equal `framework_id`,
`framework_version`, `catalogue_sha256`, `model_weights_sha256` (the `model.safetensors` hash),
preprocessing tag, bands and APS weights; any difference raises `IncompatibleSnapshots` and no
comparison is produced. The dataset hash, top-k and library versions are recorded but not checked.

For each statement ID present in both snapshots, **dDI = DI_current - DI_previous**: the change in
top-1 snapshot dissimilarity between two compatible inference runs. Positive means higher dissimilarity
in the current run, negative lower, zero unchanged. IDs in only one snapshot are listed as
`only_previous` / `only_current` without dDI. The comparison also reports the top-1 change, top-3
overlap (when both snapshots hold top-k IDs), similarity and APS changes, band transition and text-hash
change. A row is *material* when its text hash, top-1 control or band changed, or |dDI| >= 0.001.

| Situation | Result | What dDI means |
|---|---|---|
| Same text hash, compatible runs | compared | a same-text rerun; dDI shows numerical or reproducibility differences only |
| Different text hash, same ID, compatible runs | compared, row flagged `text_changed` and ranked first for review | the effect of a changed policy version under the same model and catalogue |
| Different model weights | rejected (`IncompatibleSnapshots`) | none |
| Different catalogue, framework version, preprocessing, bands or weights | rejected | none |

dDI alone does not establish temporal policy drift or compliance deterioration. A model change that
leaves `model.safetensors` unchanged (for example a different tokenizer file) is not detected by the
snapshot key; each run's own OK_FULL verification against the full manifest is the guard for that.

## 8. Outputs

Every result row carries `review_status = CANDIDATE_PENDING_HUMAN_REVIEW`. Results are candidate
mappings for a human reviewer, never a compliance determination. Logs and default exports contain
IDs, numbers and hashes only; statement text is exported only on explicit request and never inside
the repository. Each run writes a provenance record (versions, hashes, settings with paths reduced
to file names, counts, disclaimers).

CSV files are written with Python's `csv` module, so commas, quotation marks and line breaks in IDs
or text are quoted, never split. As a spreadsheet formula-injection guard, any CSV cell that starts
with `=`, `+`, `-`, `@`, tab, carriage return or line feed, or with `'`, gets one leading `'`, unless the
whole cell is a plain number (so `-0.12345678` stays numeric). Real IDs such as `R00001`, `A.5.1` or
`AC-2(1)` are unaffected. JSON exports keep the original values, and `history.read_snapshot` removes
the prefix, so snapshot round trips are exact (`csvcell.py`).

Result CSV columns (`export.RESULT_COLUMNS`; `map-multi` appends `framework_id`, `framework_version`;
queue files prepend `queue_rank`):

| Column | Meaning |
|---|---|
| `statement_id` | stable statement ID |
| `top1_control_id`, `top1_similarity` | model's first-ranked control and its cosine similarity (%.8f) |
| `tier_code`, `tier_label` | heuristic similarity review band and its display label |
| `drift_index` | DI_top1 = 1 - top1_similarity, unrounded (%.8f); legacy name |
| `aps_weight`, `aps` | tier weight and APS = drift_index x aps_weight, unrounded |
| `drift_index_r4`, `aps_r4` | legacy 4-decimal values (section 5, Rounding) |
| `top2_control_id`, `top2_similarity`, `top3_control_id`, `top3_similarity` | next candidates |
| `n_tokens_full`, `n_tokens_used`, `truncated` | statement token profile (section 6) |
| `encoder_kind`, `review_status` | encoder used; always `CANDIDATE_PENDING_HUMAN_REVIEW` |
