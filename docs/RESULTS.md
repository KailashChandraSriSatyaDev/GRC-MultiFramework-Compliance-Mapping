# Results and evidence boundaries

Figures below come from the project's evidence registry (IDs in brackets). They were produced by
the research notebooks, not by this repository, except where sections 5 and 6a say otherwise. The evidence
groups are kept separate and must not be combined or compared as if they were one experiment.

## 1. Track A: zero-shot baseline (full 206 policies, no training)

| Method | Exact | Top-3 | F1-macro |
|---|---|---|---|
| SBERT all-MiniLM-L6-v2 [A01-A03] | 28.6% (59/206) | 51.0% | 0.2294 |
| TF-IDF [A04] | 20.4% | 43.7% | 0.170 |
| BM25 [A05] | 20.4% | 45.6% | 0.168 |
| SecureBERT [A06] | 13.1% | 27.7% | 0.107 |

- Inter-annotator agreement: Cohen's kappa 0.5824 on 30 policies (18/30 exact) [A07].
- F1-optimal zero-shot threshold 0.51 [A08]. Some notebook text calls it "ROC-optimal"; it is an F1 sweep.
- Zero-shot band mix at 0.60/0.50: 58 / 76 / 72 rows [A09].

## 2. Track B-fold: fine-tuning, 5-fold evaluation (`ft_fold_*`)

- Exact match: **39.9% +/- 12.5%** (mean of folds) versus zero-shot 28.8% +/- 14.9% (k-fold mean) [B01].
- Top-3: **65.1% +/- 3.5%** versus 51.0% +/- 3.9% [B02].
- Paired t-test on fold accuracies: p = 0.0145 (t = 4.13, df = 4) [B03].
- Qualifiers: a **single unseeded run**. **Template-row effect:** 17 templated "Policy and Procedures"
  rows are matched 0/17 zero-shot and 12/17 fine-tuned; without them the gain is +6.9 percentage
  points (fold paired p = 0.133; McNemar p = 0.0024) [B03].
- The zero-shot comparator here is the k-fold mean (28.8%), not the Track A full-set figure (28.6%).

## 3. Track B-OOF: pooled out-of-fold calibration (`ft_calib_*`, a different unseeded run)

- Pooled out-of-fold exact match: 40.8% (84/206) [B04].
- AUC of top-1 similarity for correctness: zero-shot 0.5753, fine-tuned 0.6673 [B05].
- F1-optimal threshold 0.50 (F1 0.6233); the F1 curve is flat over 0.43-0.50 [B06].
- Review-band cut-offs 0.59 / 0.49 = 0.50 + 0.09 / 0.50 - 0.01 [B07]: heuristic review bands, not nested.
- Out-of-fold exact match per band: HIGH 55.1% (69 rows), REVIEW 42.6% (68), ESCALATE 24.6% (69) [B08].

Cite 39.9% / 65.1% / p = 0.0145 with Track B-fold, and 40.8% / AUC / thresholds with Track B-OOF.

## 4. In-sample figures (production model scored on its own training policies)

- In-sample exact 56.3% (116/206), top-3 82.5% [B09]. **Not a generalisation result.**
- In-sample DI_top1 / APS: mean DI 0.457, mean APS 0.585 [B13]. Not citable as evidence; with the
  default weights the APS queue order is ascending top-1 similarity.

## 5. Reproduced by this repository (local regression run, production model)

| Check | Result | Status |
|---|---|---|
| Historical band counts HIGH / REVIEW / ESCALATE | 74 / 71 / 61 | reproduced (WP2-C integration, Windows) |
| In-sample top-1 | 116/206 = 56.3% | reproduced; in-sample only |
| Batch versus single-statement mapping, 206 policies | max similarity difference 1.788e-07; no top-1 or material top-3 differences | verified (WP2-C integration, Windows) |
| Top-1 parity with the historical production mapping (R-1) | 206/206 identical; max similarity difference 2.997e-07 | reproduced (regression, Windows) |
| Top-3 parity (R-2) | no differences | reproduced (regression, Windows) |
| Band parity with the explicit legacy label mapping (R-3) | 74 / 71 / 61 in both historical files; no band-edge exceptions | reproduced (regression, Windows) |
| DI/APS arithmetic on stored values (R-4a) and recomputed 4-decimal values (R-4b) | 206/206; maximum difference 0.0 | reproduced (regression, Windows) |
| Queue order against the historical APS (R-4c) | no violations | reproduced (regression, Windows) |

## 6. Track C: external data (no gold labels)

- v2 weak labels (413 rows, 94 clusters): hit@1 zero-shot 25.2%, fine-tuned 29.1%; paired +3.9 pp
  [95% CI +0.5; +7.7] [C01]. Weak labels.
- v2 silver labels, direct (78 rows): 34.6% / 42.3% [C02]; inherited (478 rows): 30.5% / 38.7% [C03].
- v2 fine-tuned per-band hit@1 (969 rows): HIGH 50.4% (117), REVIEW 40.7% (317), ESCALATE 28.0% (535);
  AUROC 0.619 [C04]. Silver/weak labels.
- v3 real-organisation corpus: 6,118 rows in the full dataset, of which 6,104 were evaluated after
  filtering 14. Fine-tuned band mix 11.6% / 24.5% / 63.9% [C05]. **Unlabelled: distribution only,
  no accuracy claim.**

## 6a. Multi-framework external run (this repository, `map-multi`, 2026-11-19)

Separate from Track C: the same 6,104 v3 statements mapped by this repository against two catalogues.
Unlabelled, so there is no accuracy, precision, recall or F1, and no compliance or generalisation claim.

| | ISO/IEC 27001:2022 | NIST SP 800-53 Rev. 5.2.0 |
|---|---|---|
| Catalogue | 93 controls | 1,014 active controls (300 base, 714 enhancements) |
| Statements processed / rejected / excluded | 6,104 / 0 / 14 | 6,104 / 0 / 14 |
| HIGH / REVIEW / ESCALATE | 707 / 1,495 / 3,902 | 672 / 2,107 / 3,325 |
| Controls that are top-1 at least once | 93 of 93 | 740 of 1,014 |
| Truncated statements / truncated control descriptions | 0 / 0 | 0 / 4 (AC-2, CP-2, IR-8, PL-2) |
| Band basis | research-derived heuristic | ISO cut-offs reused provisionally; **not calibrated for NIST** |

- ISO band counts equal the Track C v3 figures [C05]. Against the historical v3 predictions (same
  model weights, catalogue, text and scoring rules), all 6,104 top-1 controls and bands are unchanged
  and max |dDI| = 5.01e-05, consistent with 4-decimal rounding of the historical file: a same-text
  rerun, not a change in any policy.
- NIST has no previous run; this run is its baseline snapshot.
- ISO and NIST similarity and DI distributions are not directly comparable (catalogue size and
  structure differ). Cross-framework figures describe candidate-mapping differences only.
- A comparison against LLM-generated silver labels for v3 [C06] is unverified and not reported.

## 7. Not used

Hard-negative fine-tuning [B10], the TF-IDF+SBERT ensemble [B11, retired], the single held-out
split [B12, rejected design], the out-of-fold DI diagnostic using gold labels [B14, diagnostic only;
it uses evaluation labels and is not DI_mapped, which needs an independently assigned control]
and an unsourced "80% / 8.8%" band claim [X01, no source] are not used as evidence.
