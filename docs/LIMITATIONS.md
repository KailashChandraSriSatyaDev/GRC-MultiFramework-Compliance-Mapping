# Limitations

1. **Small, single-source training set.** 206 labelled policies for 93 controls. Most controls have
   few examples; some have none.
2. **Unseeded fine-tuning.** The 5-fold evaluation and the out-of-fold calibration are two
   different unseeded runs; neither has been replicated with fixed seeds.
3. **Template-row effect.** 17 templated rows contribute much of the fine-tuning gain (see RESULTS.md, Track B-fold).
4. **Thresholds are heuristic.** 0.59 / 0.49 come from an F1 sweep plus fixed offsets on the same
   out-of-fold scores, without nesting. Cosine similarity is not a probability.
5. **DI is a top-1 snapshot dissimilarity.** DI_top1 = 1 - similarity to the model's own top-1
   candidate in one run (legacy field name `drift_index`). It is not temporal policy drift, not
   evidence of policy change, control failure or non-compliance, and not a calibrated risk probability.
   DI_mapped (against an independently assigned control) is not implemented: no input carries an
   assigned control, and neither predicted controls nor evaluation labels may substitute for one.
6. **APS adds no ranking information beyond similarity.** APS = DI_top1 x tier weight depends on the
   top-1 similarity alone. With weights ordered HIGH <= REVIEW <= ESCALATE (the defaults) its queue order
   is exactly ascending top-1 similarity; other positive weights are accepted by the configuration and
   break that equivalence. It is a heuristic audit-review priority score: not a validated or calibrated
   risk score and not a measure of control criticality.
7. **In-sample production model.** The production model was trained on all 206 policies, so its
   scores on those policies are in-sample.
8. **External data has no gold labels.** Track C uses weak/silver labels (v2) or none (v3).
9. **Truncation.** Statements and control descriptions longer than 256 tokens (special tokens included) are truncated by the
   model; this is reported per item, not prevented.
10. **Band decision precision.** Bands are decided on the float64 value of the float32 similarity;
    the research notebook compared in float32, so a value exactly at float32(0.59) or float32(0.49)
    could fall on the other side. The regression check allows a 1e-6 band-edge exception and lists it.
11. **Batch versus single.** The mapping code is shared; with the real model, batch padding inside
    the encoder can move similarities by about 1e-7 (observed maximum 1.788e-07 on 206 policies).
12. **Offline guard scope.** The socket guard is application-level only.
13. **Public demonstration.** The demo uses a hash-seeded fake encoder; its scores carry no meaning.
    Reviewers without the private model can run the unit tests and the demo only.
14. **Human review required.** No output is a compliance determination.
15. **Other frameworks.** The model was fine-tuned on ISO/IEC 27001 texts only; mappings to other
    catalogues (for example NIST SP 800-53) are out-of-domain for the fine-tuning. Reusing the ISO
    cut-offs 0.59 / 0.49 and weights for another framework is a provisional triage setting, not a
    calibrated threshold. Similarity, DI and band shares are not comparable across catalogues of
    different size and structure (a larger catalogue raises the expected top-1 maximum and narrows
    margins). Cross-framework candidates are not a crosswalk, an equivalence or coverage evidence.
16. **dDI is a between-run difference.** dDI = DI_current - DI_previous is produced only for
    compatible snapshots (same framework and version, catalogue hash, model-weights hash,
    preprocessing, bands and weights). It reflects rerun noise when the text hash is unchanged and the
    effect of a changed policy version when it differs; it does not by itself establish temporal drift
    or compliance deterioration. The snapshot key checks only the `model.safetensors` hash, not other
    model files, the dataset hash or top-k.
17. **Truncation has two sides.** Statement truncation and control-description truncation are reported
    separately; a truncated control is scored on its first 256 tokens only.
