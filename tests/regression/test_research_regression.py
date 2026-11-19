"""Regression tests R-1 ... R-7 against the historical research outputs (local-only fixtures).

Needs GRC_CONFIG. R-6 and R-7 need only the local research data; R-1 ... R-5 also need the two
historical CSVs staged (hash-verified) in paths.regression_dir. Fixture-dependent tests SKIP with
an explicit reason until the fixtures are staged. Hashes come from the local expectations file,
never from this repository. Only IDs and numbers are read; no text is printed.
"""

import ast
import csv
import io
import json
import os

import numpy as np
import pytest

if not os.environ.get("GRC_CONFIG"):
    pytest.skip("GRC_CONFIG not set: regression tests need the local configuration", allow_module_level=True)

from grc_mapper.config import load_settings  # noqa: E402
from grc_mapper.data import load_control_catalog, load_statements  # noqa: E402
from grc_mapper.integrity import sha256_file  # noqa: E402
from grc_mapper.pipeline import build_control_index, load_encoder, map_single, map_statements  # noqa: E402
from grc_mapper.scoring import aps_weight, priority_queue, round4, tier_code  # noqa: E402

S = load_settings()
COLS = S.columns
EXP = json.loads(S.require_path("regression_expectations").read_text(encoding="utf-8"))
TOL = EXP["tolerances"]
MAPPING_CSV = "grc_mapping_results_finetuned_production.csv"
DI_APS_CSV = "grc_full_results_DI_APS.csv"

LEGACY_TIER_TO_CODE = {
    "High Confidence": "HIGH",                    # expected 74
    "Requires Human Review": "REVIEW",            # expected 71
    "Low Confidence — Escalate": "ESCALATE",  # expected 61 (U+2014 em dash)
}
#: Hyphen variant used by the external-data notebooks. Documented only; NOT part of R-3.
TRACK_C_LEGACY_LABELS = {"Low Confidence - Escalate": "ESCALATE"}


def _fixture(name):
    d = S.path("regression_dir")
    if d is None or not (d / name).is_file():
        pytest.skip(f"regression fixture {name} not staged in paths.regression_dir "
                    "(run the reviewed staging script after approval)")
    p = d / name
    exp = EXP["fixtures"][name]
    assert p.stat().st_size == exp["bytes"] and sha256_file(p) == exp["sha256"], f"{name} hash mismatch"
    rows = list(csv.DictReader(io.StringIO(p.read_bytes().decode("utf-8-sig"), newline="")))
    return {r["Policy_ID"]: r for r in rows}


@pytest.fixture(scope="module")
def ours():
    enc = load_encoder(S)
    cat = load_control_catalog(S.require_path("controls_csv"), id_column=COLS["control_id"],
                               text_column=COLS["control_text"], expected_count=EXP["data"]["controls"]["rows"])
    stmts, labels = load_statements(S.require_path("policies_csv"), id_column=COLS["statement_id"],
                                    text_column=COLS["statement_text"], label_column=COLS["label"],
                                    known_controls=cat, expected_count=EXP["data"]["policies"]["rows"])
    idx = build_control_index(cat, enc, S)
    rows = map_statements(stmts, idx, enc, S)
    return {"enc": enc, "idx": idx, "stmts": stmts, "labels": labels, "rows": {r.statement_id: r for r in rows},
            "order": [r.statement_id for r in rows]}


def test_r1_top1_parity(ours):
    hist = _fixture(MAPPING_CSV)
    assert set(hist) == set(ours["rows"])
    same = [pid for pid, h in hist.items() if h["Predicted_Control_ID"] == ours["rows"][pid].top1_control_id]
    dmax = max(abs(float(h["Similarity_Score"]) - ours["rows"][pid].top1_similarity) for pid, h in hist.items())
    print(f"R-1: top-1 identical {len(same)}/206; max |dsim| {dmax:.3e}")
    assert len(same) == 206 and dmax <= TOL["r1_sim_abs"]


def test_r2_top3_parity(ours):
    hist = _fixture(MAPPING_CSV)
    diffs, near = [], []
    for pid, h in hist.items():
        r = ours["rows"][pid]
        want = list(ast.literal_eval(h["Top3_Controls"]))
        got = [c.control_id for c in r.candidates[:3]]
        if got != want:
            gaps = [r.candidates[i].similarity - r.candidates[i + 1].similarity for i in range(2)]
            (near if min(gaps) < TOL["r2_topk_gap"] else diffs).append(pid)
    print(f"R-2: top-3 differences {diffs}; near-tie differences (listed, allowed) {near}")
    assert not diffs


def test_r3_tier_parity_explicit_mapping(ours):
    for name in (MAPPING_CSV, DI_APS_CSV):
        hist = _fixture(name)
        labels = {h["Confidence_Flag"] for h in hist.values()}
        assert labels <= set(LEGACY_TIER_TO_CODE), f"{name}: unmapped legacy labels {sorted(labels - set(LEGACY_TIER_TO_CODE))}"
        counts = {lab: sum(h["Confidence_Flag"] == lab for h in hist.values()) for lab in LEGACY_TIER_TO_CODE}
        assert counts == EXP["legacy_tier_label_counts"], name
        edge = []
        for pid, h in hist.items():
            code = LEGACY_TIER_TO_CODE[h["Confidence_Flag"]]
            assert code == tier_code(float(h["Similarity_Score"])), (name, pid)
            ours_sim = ours["rows"][pid].top1_similarity
            if code != ours["rows"][pid].tier_code:
                assert min(abs(ours_sim - 0.59), abs(ours_sim - 0.49)) <= TOL["r3_band_edge"], (name, pid)
                edge.append(pid)
        print(f"R-3 {name}: counts {counts}; band-edge exceptions {edge}")


def test_r4a_stored_arithmetic(ours):
    hist = _fixture(DI_APS_CSV)
    bad = []
    for pid, h in hist.items():
        s, w = float(h["Similarity_Score"]), float(h["Confidence_Weight"])
        di4 = round4(1.0 - s)
        code = LEGACY_TIER_TO_CODE[h["Confidence_Flag"]]
        ok = (abs(di4 - float(h["Drift_Index"])) <= 1e-12 and abs(round4(di4 * w) - float(h["APS"])) <= 1e-12
              and aps_weight(code) == w)
        if not ok:
            bad.append(pid)
    print(f"R-4a: stored DI/APS arithmetic reproduced {206 - len(bad)}/206")
    assert not bad


def test_r4b_recomputed_r4_values(ours):
    hist = _fixture(DI_APS_CSV)
    d_di = max(abs(ours["rows"][p].drift_index_r4 - float(h["Drift_Index"])) for p, h in hist.items())
    d_aps = max(abs(ours["rows"][p].aps_r4 - float(h["APS"])) for p, h in hist.items())
    print(f"R-4b: max |d DI_r4| {d_di:.1e}; max |d APS_r4| {d_aps:.1e}")
    assert d_di <= TOL["r4b_di_r4_abs"] + 1e-12 and d_aps <= TOL["r4b_aps_r4_abs"] + 1e-12


def test_r4c_queue_order(ours):
    hist = _fixture(DI_APS_CSV)
    q = [r.statement_id for r in priority_queue(list(ours["rows"].values()))]
    hist_aps = [float(hist[p]["APS"]) for p in q]
    violations = [(q[i], q[i + 1]) for i in range(len(q) - 1) if hist_aps[i + 1] > hist_aps[i]]
    print(f"R-4c: queue order violations vs historical APS (strictly greater later) {violations}")
    assert not violations


def test_r5_batch_versus_single(ours):
    enc, idx = ours["enc"], ours["idx"]
    dmax, top1 = 0.0, []
    for s in ours["stmts"]:
        r = map_single(s.text, idx, enc, S, statement_id=s.statement_id)
        b = ours["rows"][s.statement_id]
        dmax = max(dmax, max(abs(x.similarity - y.similarity) for x, y in zip(r.candidates, b.candidates)))
        if r.top1_control_id != b.top1_control_id:
            top1.append(s.statement_id)
    print(f"R-5: batch vs single max |dsim| {dmax:.3e}; top-1 differences {top1}")
    assert dmax <= TOL["r5_batch_single_sim_abs"] and not top1


def test_r6_in_sample_top1_reproduction_not_generalisation(ours):
    correct = sum(ours["rows"][sid].top1_control_id == lab for sid, lab in ours["labels"].items())
    pct = round(100.0 * correct / len(ours["labels"]), 1)
    print(f"R-6: in-sample top-1 {correct}/{len(ours['labels'])} = {pct}% (in-sample reproduction, not generalisation)")
    assert correct == EXP["in_sample_top1"]["correct"] and pct == EXP["in_sample_top1"]["accuracy_pct_1dp"]


def test_r7_research_inputs():
    assert sha256_file(S.require_path("controls_csv")) == EXP["data"]["controls"]["sha256"]
    assert sha256_file(S.require_path("policies_csv")) == EXP["data"]["policies"]["sha256"]
    d = S.path("regression_dir")
    staged = [n for n in (MAPPING_CSV, DI_APS_CSV) if d is not None and (d / n).is_file()]
    for n in staged:
        assert sha256_file(d / n) == EXP["fixtures"][n]["sha256"], n
    print(f"R-7: data hashes OK; fixtures staged and verified: {staged}")
