"""Integration tests with the REAL local production model and research data (authoritative run: grc-mvp).

Needs GRC_CONFIG (local TOML). Prints IDs and counts only. The in-sample top-1 check reproduces the
recorded in-sample figure; it is not a generalisation result.
"""

import json
import logging
import os
import shutil
import socket
import subprocess
import sys

import numpy as np
import pytest

if not os.environ.get("GRC_CONFIG"):
    pytest.skip("GRC_CONFIG not set: integration tests need the local configuration", allow_module_level=True)

from _support import child_env  # noqa: E402
from grc_mapper.config import load_settings  # noqa: E402
from grc_mapper.data import load_control_catalog, load_statements  # noqa: E402
from grc_mapper.integrity import sha256_file  # noqa: E402
from grc_mapper.model import profile_texts  # noqa: E402
from grc_mapper.offline import OFFLINE_ENV  # noqa: E402
from grc_mapper.pipeline import build_control_index, load_encoder, map_single, map_statements  # noqa: E402
from grc_mapper.scoring import priority_queue  # noqa: E402
from grc_mapper.text import TokenProfile  # noqa: E402

S = load_settings()
COLS = S.columns
EXP = json.loads(S.require_path("regression_expectations").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def world():
    enc = load_encoder(S)
    cat = load_control_catalog(S.require_path("controls_csv"), id_column=COLS["control_id"],
                               text_column=COLS["control_text"], expected_count=EXP["data"]["controls"]["rows"])
    stmts, labels = load_statements(S.require_path("policies_csv"), id_column=COLS["statement_id"],
                                    text_column=COLS["statement_text"], label_column=COLS["label"],
                                    known_controls=cat, expected_count=EXP["data"]["policies"]["rows"])
    idx = build_control_index(cat, enc, S)
    batch = map_statements(stmts, idx, enc, S)
    return enc, cat, stmts, labels, idx, batch


# ----------------------------------------------------------------------------- model (I-MD, I-TK)

def test_i_md_1_ok_full_offline_contract(world):
    enc = world[0]
    rep = enc.report
    assert rep.status == "OK_FULL" and rep.runtime_files_verified == 8
    assert dict(rep.non_runtime) == {"README.md": "verified",
                                     "eval/similarity_evaluation_train_dev_results.csv": "verified"}
    assert enc.embedding_dim == 384 and enc.max_seq_length == 256
    assert all(os.environ.get(k) == v for k, v in OFFLINE_ENV.items())
    assert socket.create_connection.__name__ != "_blocked"


def test_i_md_2_tampered_copy_fails_before_import(tmp_path):
    copy = tmp_path / "model_copy"
    shutil.copytree(S.require_path("model_dir"), copy)
    cfg = copy / "config.json"
    data = bytearray(cfg.read_bytes())
    data[-2] = ord(" ") if data[-2] != ord(" ") else ord("\t")
    cfg.write_bytes(bytes(data))
    code = (
        "import sys\nfrom pathlib import Path\nfrom grc_mapper.model import load_verified_model\n"
        "from grc_mapper.errors import IntegrityError\ntry:\n"
        f"    load_verified_model(Path({str(copy)!r}), runtime_manifest_path=Path({str(S.require_path('model_manifest'))!r}),"
        f" full_manifest_path=Path({str(S.require_path('model_full_manifest'))!r}))\n"
        "    print('LOADED'); sys.exit(1)\nexcept IntegrityError:\n"
        "    bad=[m for m in ('sentence_transformers','transformers','torch') if m in sys.modules]\n"
        "    print('INTEGRITY_ERROR imported=' + ','.join(bad)); sys.exit(0 if not bad else 2)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=child_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "INTEGRITY_ERROR imported=" in r.stdout


def test_i_md_4_repeat_encode_deterministic(world):
    enc = world[0]
    probes = ["access review procedure", "backup schedule for servers", "incident reporting"]
    a, b = enc.encode(probes, batch_size=3), enc.encode(probes, batch_size=3)
    assert float(np.max(np.abs(a - b))) <= 1e-6
    assert float(np.max(np.abs(np.linalg.norm(a, axis=1) - 1))) <= 1e-3


def _probe_word(tok):
    for w in ("control", "access", "policy", "system", "data"):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            continue
        wid = ids[0]
        if all(tok(" ".join([w] * n), add_special_tokens=False)["input_ids"] == [wid] * n for n in (1, 2, 254, 255)):
            return w, wid
    pytest.fail("no single-token probe word found in the candidate list")


def test_i_tk_1_token_ids_and_truncation(world):
    enc = world[0]
    st, tok = enc._st, enc._st.tokenizer
    assert tok.num_special_tokens_to_add(pair=False) == 2
    cls, sep = tok.cls_token_id, tok.sep_token_id
    assert cls is not None and sep is not None and st.max_seq_length == 256
    w, wid = _probe_word(tok)
    print(f"probe word: {w!r} id={wid}")
    cases = {
        1: ([cls, wid, sep], [cls, wid, sep], TokenProfile(3, 3, False)),
        254: ([cls] + [wid] * 254 + [sep], None, TokenProfile(256, 256, False)),
        255: ([cls] + [wid] * 255 + [sep], [cls] + [wid] * 254 + [sep], TokenProfile(257, 256, True)),
        600: ([cls] + [wid] * 600 + [sep], [cls] + [wid] * 254 + [sep], TokenProfile(602, 256, True)),
    }
    for n, (exp_full, exp_used, exp_prof) in cases.items():
        t = " ".join([w] * n)
        saved = str(t)
        assert tok(t, add_special_tokens=True, truncation=False, verbose=False)["input_ids"] == exp_full, n
        feats = st.tokenize([t])
        used = feats["input_ids"][0].tolist()
        assert used == (exp_used if exp_used is not None else exp_full), n
        assert feats["attention_mask"][0].tolist() == [1] * len(used)
        assert used[0] == cls and used[-1] == sep and sum(1 for i in used if i in (cls, sep)) == 2
        assert enc.token_profile(t) == exp_prof and t == saved


def test_i_tk_2_profiles_research_controls_and_policies(world):
    enc, cat, stmts, _, idx, _ = world
    sp = profile_texts(enc, [s.statement_id for s in stmts], [s.text for s in stmts], kind="statement")
    assert len(idx.token_profiles) == 93 and len(sp) == 206
    ts = [s.statement_id for s, p in zip(stmts, sp) if p.truncated]
    print(f"controls truncated: {list(idx.truncated_control_ids)}; statements truncated: {ts}")


# ----------------------------------------------------------------------------- pipeline (C1-C6)

def test_c1_inputs_and_index(world):
    enc, cat, stmts, labels, idx, _ = world
    assert sha256_file(S.require_path("controls_csv")) == EXP["data"]["controls"]["sha256"]
    assert sha256_file(S.require_path("policies_csv")) == EXP["data"]["policies"]["sha256"]
    assert idx.model_fingerprint == sha256_file(S.require_path("model_manifest"))
    assert idx.embeddings.shape == (93, 384) and idx.embeddings.dtype == np.float32


def test_c2_tiers_and_in_sample_reproduction(world):
    _, _, stmts, labels, _, batch = world
    assert len(batch) == 206 and all(r.review_status == "CANDIDATE_PENDING_HUMAN_REVIEW" for r in batch)
    counts = {t: sum(r.tier_code == t for r in batch) for t in ("HIGH", "REVIEW", "ESCALATE")}
    legacy = EXP["legacy_tier_label_counts"]
    expected = {"HIGH": legacy["High Confidence"], "REVIEW": legacy["Requires Human Review"],
                "ESCALATE": legacy["Low Confidence — Escalate"]}
    print(f"tier counts: {counts} (historical {expected})")
    assert counts == expected
    correct = sum(r.top1_control_id == labels[r.statement_id] for r in batch)
    print(f"in-sample top-1: {correct}/206 (in-sample only; not generalisation)")
    assert correct == EXP["in_sample_top1"]["correct"]


def test_c3_batch_versus_single(world):
    enc, _, stmts, _, idx, batch = world
    max_d, top1_diff, top3_diff, tier_diff = 0.0, [], [], []
    for s, b in zip(stmts, batch):
        r = map_single(s.text, idx, enc, S, statement_id=s.statement_id)
        max_d = max(max_d, max(abs(x.similarity - y.similarity) for x, y in zip(r.candidates, b.candidates)))
        if r.top1_control_id != b.top1_control_id:
            top1_diff.append(s.statement_id)
        if [c.control_id for c in r.candidates] != [c.control_id for c in b.candidates]:
            gaps = [b.candidates[i].similarity - b.candidates[i + 1].similarity for i in range(len(b.candidates) - 1)]
            if min(gaps) >= 1e-5:
                top3_diff.append(s.statement_id)
        if r.tier_code != b.tier_code and min(abs(b.top1_similarity - 0.59), abs(b.top1_similarity - 0.49)) > 1e-6:
            tier_diff.append(s.statement_id)
    print(f"batch vs single: max |dsim| = {max_d:.3e}; top-1 diffs {top1_diff}; top-3 diffs {top3_diff}; "
          f"tier diffs {tier_diff}")
    assert max_d <= 1e-5 and not top1_diff and not top3_diff and not tier_diff


def test_c4_batch_size_invariance(world):
    enc, cat, stmts, _, _, batch = world
    for bs in (1, 206):
        st = load_settings({"batch_size": bs})
        other = map_statements(stmts, build_control_index(cat, enc, st), enc, st)
        d = max(abs(a.top1_similarity - b.top1_similarity) for a, b in zip(batch, other))
        same = sum(a.top1_control_id == b.top1_control_id for a, b in zip(batch, other))
        print(f"batch_size {bs} vs 32: max |dsim top1| = {d:.3e}; top-1 identical {same}/206")
        assert d <= 1e-5 and same == 206


def test_c5_rerun_identical_and_queue(world):
    enc, _, stmts, _, idx, batch = world
    assert map_statements(stmts, idx, enc, S) == batch
    sims = [r.top1_similarity for r in priority_queue(batch)]
    assert sims == sorted(sims)


def test_c6_no_text_in_logs(world, caplog):
    enc, cat, stmts, _, idx, _ = world
    with caplog.at_level(logging.DEBUG):
        build_control_index(cat, enc, S)
        map_statements(stmts, idx, enc, S)
    logs = "\n".join(r.getMessage() for r in caplog.records)
    probes = [s.text.strip()[:25] for s in stmts] + [t.strip()[:25] for t in cat.texts]
    assert not [p for p in probes if len(p) >= 10 and p in logs]
