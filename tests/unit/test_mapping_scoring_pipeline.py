"""Unit tests: config, mapping, scoring, pipeline (isolated test encoder). No model, no private data."""

import itertools
import logging
import math
import os
import random

import numpy as np
import pytest

from grc_mapper import config as C
from grc_mapper.config import load_settings, validate_bands, validate_weights
from grc_mapper.data import ControlCatalog, Statement, load_control_catalog
from grc_mapper.errors import ConfigError, GrcError, InputValidationError, IntegrityError
from grc_mapper.mapping import Candidate, ranking_order, similarity_matrix, top_k, validate_embeddings
from grc_mapper.scoring import (
    ScoredMapping, aps, aps_weight, drift_index, priority_queue, queue_key, round4, score_row, tier_code,
)
from grc_mapper.text import TokenProfile

pytestmark = pytest.mark.unit
SECRET = "ZQX-SENTINEL wording that must never leak"
PROF = TokenProfile(5, 5, False)


# =============================================================== config

def test_research_constants():
    assert (C.TIER_HIGH_MIN, C.TIER_REVIEW_MIN) == (0.59, 0.49)
    assert dict(C.APS_WEIGHTS) == {"HIGH": 1.00, "REVIEW": 1.25, "ESCALATE": 1.50}
    assert C.REVIEW_STATUS == "CANDIDATE_PENDING_HUMAN_REVIEW"
    assert C.BAND_TERM == "heuristic similarity review band"
    assert all("confidence" not in v.lower() for v in C.TIER_DISPLAY.values())
    assert all(v.isascii() for v in C.TIER_DISPLAY.values())
    from grc_mapper import model
    assert (model.EXPECTED_EMBED_DIM, model.EXPECTED_MAX_SEQ) == (C.EXPECTED_EMBED_DIM, C.EXPECTED_MAX_SEQ)


def test_defaults_have_no_paths_except_repo_manifest():
    s = load_settings(env={})
    assert all(v is None for k, v in s.paths.items() if k != "model_manifest")
    assert (s.device, s.batch_size, s.top_k, s.bands_are_default, s.weights_are_default) == ("cpu", 32, 3, True, True)
    with pytest.raises(ConfigError, match="paths.model_dir"):
        s.require_path("model_dir")


def _toml(tmp_path, body, name="c.toml"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_resolution_order(tmp_path):
    cfg = _toml(tmp_path, "[paths]\nmodel_dir = 'from_toml'\ncontrols_csv = 'c_toml'\n[runtime]\ndevice = 'cpu'\nbatch_size = 8\n")
    env = {"GRC_CONFIG": str(cfg), "GRC_MODEL_DIR": "from_env", "GRC_DEVICE": "cuda"}
    s = load_settings({"model_dir": "from_cli"}, env=env)
    assert s.require_path("model_dir").name == "from_cli"
    assert s.require_path("controls_csv").name == "c_toml"
    assert s.device == "cuda" and s.batch_size == 8
    s2 = load_settings(env=env)
    assert s2.require_path("model_dir").name == "from_env"
    s3 = load_settings({"config": str(cfg)}, env={})
    assert s3.require_path("model_dir").name == "from_toml" and s3.config_file == cfg


def test_path_expansion(tmp_path):
    s = load_settings({"model_dir": "~/m", "output_dir": "$GRC_TEST_ROOT/out"}, env={})
    assert not str(s.require_path("model_dir")).startswith("~")
    os.environ["GRC_TEST_ROOT"] = str(tmp_path)
    try:
        s = load_settings({"output_dir": "$GRC_TEST_ROOT/out"}, env={})
        assert s.require_path("output_dir") == tmp_path / "out"
    finally:
        del os.environ["GRC_TEST_ROOT"]


@pytest.mark.parametrize("body,msg", [
    ("[pathz]\nx = 1\n", "unknown config section"),
    ("[paths]\nmodel_dirr = 'x'\n", "unknown keys"),
    ("[paths]\nmodel_dir = '<MODEL_DIR>'\n", "template placeholder"),
    ("[paths]\nmodel_dir = ''\n", "is empty"),
    ("[runtime]\nbatch_size = 0\n", "batch_size"),
    ("[runtime]\ntop_k = true\n", "top_k"),
    ("[runtime]\ndevice = 'tpu'\n", "device"),
    ("[bands]\nhigh_min = 0.49\nreview_min = 0.49\n", "review_min < high_min"),
    ("[bands]\nhigh_min = 1.5\n", "review_min < high_min"),
    ("[bands]\nhigh_min = true\n", "finite number"),
    ("[bands]\nhigh_min = nan\n", "finite number"),
    ("[aps_weights]\nHIGH = 1.0\nREVIEW = 1.25\n", "exactly the keys"),
    ("[aps_weights]\nHIGH = 0\nREVIEW = 1.25\nESCALATE = 1.5\n", "> 0"),
    ("[columns]\ncontrol_id = ''\n", "non-empty string"),
    ("[scanner]\nidentity_tokens = 'x'\n", "identity_tokens"),
    ("not = [valid", "not valid UTF-8 TOML"),
])
def test_config_rejects(tmp_path, body, msg):
    with pytest.raises(ConfigError, match=msg):
        load_settings({"config": str(_toml(tmp_path, body))}, env={})


def test_missing_config_and_unknown_cli(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_settings({"config": str(tmp_path / "nope.toml")}, env={})
    with pytest.raises(ConfigError, match="unknown CLI overrides"):
        load_settings({"thresholds": 1}, env={})


def test_example_template_is_rejected_until_filled():
    with pytest.raises(ConfigError, match="placeholder"):
        load_settings({"config": str(C.REPO_ROOT / "config" / "example.local.toml")}, env={})


def test_overrides_flagged(tmp_path):
    cfg = _toml(tmp_path, "[bands]\nhigh_min = 0.6\nreview_min = 0.5\n[aps_weights]\nHIGH = 1\nREVIEW = 2\nESCALATE = 3\n")
    s = load_settings({"config": str(cfg)}, env={})
    assert (s.high_min, s.review_min, s.bands_are_default, s.weights_are_default) == (0.6, 0.5, False, False)
    red = s.redacted()
    assert red["bands"]["default"] is False and red["config_file"] == "c.toml"


def test_redacted_paths_are_basenames(tmp_path):
    s = load_settings({"model_dir": str(tmp_path / "deep" / "model")}, env={})
    assert s.redacted()["paths"]["model_dir"] == "model"


def test_validators_direct():
    assert validate_bands(0.59, 0.49) == (0.59, 0.49)
    assert dict(validate_weights({"HIGH": 1, "REVIEW": 1.25, "ESCALATE": 1.5})) == {"HIGH": 1.0, "REVIEW": 1.25, "ESCALATE": 1.5}


def test_real_local_toml_shape_is_accepted(tmp_path):
    body = (C.REPO_ROOT.parent / "local_config" / "grc_local.toml")
    if not body.is_file():
        pytest.skip("local config not present in this workspace")
    s = load_settings({"config": str(body)}, env={})
    assert s.columns["statement_id"] == "Policy_ID" and s.bands_are_default


# =============================================================== mapping

def _unit(rows):
    a = np.asarray(rows, dtype=np.float64)
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


def test_validate_embeddings():
    ok = _unit([[1, 0], [0, 1]])
    assert validate_embeddings(ok, n=2, dim=2).dtype == np.float32
    for bad, n, msg in [(ok[:1], 2, "shape"), (ok * 2, 2, "not unit-normalised"),
                        (np.array([[np.nan, 1.0]]), 1, "NaN"), (np.array([[1, 0]]), 1, "floating point")]:
        with pytest.raises(GrcError, match=msg):
            validate_embeddings(bad, n=n, dim=2)


def test_similarity_matrix_equals_cosine():
    rng = np.random.default_rng(0)
    s, c = _unit(rng.standard_normal((5, 8))), _unit(rng.standard_normal((7, 8)))
    sims = similarity_matrix(s, c)
    ref = (s.astype(np.float64) @ c.astype(np.float64).T) / np.outer(np.linalg.norm(s, axis=1), np.linalg.norm(c, axis=1))
    assert sims.dtype == np.float32 and sims.shape == (5, 7) and float(np.max(np.abs(sims - ref))) <= 1e-6
    with pytest.raises(GrcError, match="dimensions"):
        similarity_matrix(s, c[:, :4])
    with pytest.raises(GrcError, match="outside"):
        similarity_matrix(s * 2, c)


def test_similarity_independent_of_batch_composition():
    rng = np.random.default_rng(11)
    s, c = _unit(rng.standard_normal((257, 384))), _unit(rng.standard_normal((93, 384)))
    full = similarity_matrix(s, c)
    for i in range(0, 257, 16):
        assert np.array_equal(similarity_matrix(s[i:i + 1], c)[0], full[i])
    assert np.array_equal(similarity_matrix(s[5:9], c), full[5:9])


def test_top_k_order_bounds_and_ties():
    ids = ("A.5.1", "A.5.2", "A.5.10", "A.8.1")
    row = np.array([0.3, 0.7, 0.7, 0.1], dtype=np.float32)
    c = top_k(row, 3, ids)
    assert [x.control_id for x in c] == ["A.5.2", "A.5.10", "A.5.1"]  # tie -> lower catalogue position
    assert [x.rank for x in c] == [1, 2, 3] and isinstance(c[0].similarity, float)
    assert [x.control_id for x in top_k(row, 4, ids)][-1] == "A.8.1"
    for k in (0, 5, True, 1.0):
        with pytest.raises(ConfigError):
            top_k(row, k, ids)
    with pytest.raises(GrcError):
        top_k(row[:3], 1, ids)
    with pytest.raises(GrcError):
        top_k(np.array([0.1, np.nan, 0.2, 0.3]), 1, ids)


def test_ranking_order_matches_bruteforce():
    rng = np.random.default_rng(3)
    for _ in range(50):
        row = rng.choice(np.array([0.1, 0.2, 0.5, 0.5, 0.9], dtype=np.float32), size=12)
        expect = sorted(range(12), key=lambda i: (-float(row[i]), i))
        assert ranking_order(row).tolist() == expect


# =============================================================== scoring

@pytest.mark.parametrize("sim,tier", [
    (0.59, "HIGH"), (0.589999, "REVIEW"), (0.49, "REVIEW"), (0.489999, "ESCALATE"),
    (1.0, "HIGH"), (-1.0, "ESCALATE"), (0.75, "HIGH"), (0.0, "ESCALATE"),
])
def test_tier_boundaries(sim, tier):
    assert tier_code(sim) == tier


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 1.1, -1.1, True, "0.5", None])
def test_tier_rejects(bad):
    with pytest.raises(GrcError):
        tier_code(bad)


def test_tier_custom_bands():
    assert tier_code(0.55, 0.6, 0.5) == "REVIEW"
    with pytest.raises(ConfigError):
        tier_code(0.5, 0.4, 0.4)


def test_di_aps_unrounded():
    for s in np.linspace(-1, 1, 41):
        assert drift_index(float(s)) == 1.0 - float(s)
    assert aps_weight("HIGH") == 1.0 and aps_weight("REVIEW") == 1.25 and aps_weight("ESCALATE") == 1.5
    assert aps(0.5857, 1.5) == 0.5857 * 1.5
    with pytest.raises(ConfigError):
        aps_weight("MEDIUM")


def test_round4_numpy_semantics_and_sequential_aps():
    assert round4(0.87855) == 0.8786       # numpy; Python's round(0.87855, 4) gives 0.8785
    assert round(0.8785499999999999, 4) == 0.8785
    sim = 0.41425902                        # documented historical row: DI4 0.5857, w 1.5, APS4 0.8786
    r = score_row("P1", (Candidate(1, "A.6.1", sim),), PROF, "TEST", load_settings(env={}))
    assert (r.tier_code, r.drift_index_r4, r.aps_weight, r.aps_r4) == ("ESCALATE", 0.5857, 1.5, 0.8786)
    assert r.drift_index == 1.0 - sim and r.aps == (1.0 - sim) * 1.5   # unrounded kept separately


def test_score_row_fields(settings):
    cands = (Candidate(1, "A.5.1", 0.66733253), Candidate(2, "A.5.36", 0.5), Candidate(3, "A.5.18", 0.4))
    r = score_row("P1", cands, PROF, "TEST", settings)
    assert (r.top1_control_id, r.tier_code, r.tier_label) == ("A.5.1", "HIGH", "High similarity - candidate mapping")
    assert r.review_status == "CANDIDATE_PENDING_HUMAN_REVIEW" and r.encoder_kind == "TEST"
    assert r.candidates == cands and r.token_profile is PROF
    with pytest.raises(GrcError):
        score_row("P1", (), PROF, "TEST", settings)


def _row(sid, sim, settings):
    return score_row(sid, (Candidate(1, "A.5.1", sim),), PROF, "TEST", settings)


def test_queue_order_equivalent_to_ascending_similarity(settings):
    rng = random.Random(7)
    sims = [rng.uniform(-0.2, 0.95) for _ in range(300)] + [0.59, 0.49, 0.5899999, 0.4899999]
    rows = [_row(f"S{i:04d}", s, settings) for i, s in enumerate(sims)]
    q = priority_queue(rows)
    assert [r.top1_similarity for r in q] == sorted(sims)


def test_queue_tie_breaks(settings):
    rows = [_row("S2", 0.3, settings), _row("S1", 0.3, settings), _row("S3", 0.8, settings)]
    assert [r.statement_id for r in priority_queue(rows)] == ["S1", "S2", "S3"]
    assert queue_key(rows[0]) == (-rows[0].aps, 0.3, "S2")
    with pytest.raises(GrcError):
        priority_queue([rows[0], rows[0]])


def test_queue_uses_unrounded_aps(settings):
    a, b = _row("A", 0.41425902, settings), _row("B", 0.41425899, settings)
    assert a.aps_r4 == b.aps_r4 and [r.statement_id for r in priority_queue([a, b])] == ["B", "A"]


# =============================================================== pipeline

from grc_mapper.pipeline import ControlIndex, build_control_index, map_single, map_statements  # noqa: E402


def _catalog(n=9, dup_text=None):
    ids = [f"A.5.{i}" for i in range(1, n + 1)]
    texts = [f"control {i} about topic {i % 4} and area {i % 3}" for i in range(1, n + 1)]
    if dup_text is not None:
        for i in dup_text:
            texts[i] = "identical control description"
    order = sorted(range(n), key=lambda k: [int(x) if x.isdigit() else x for x in ids[k].split(".")])
    return ControlCatalog(tuple(ids[k] for k in order), tuple(texts[k] for k in order), "0" * 64, "synthetic")


STMTS = [Statement(f"P{i:03d}", f"statement {i} on topic {i % 5} with words " + "w " * (i % 15)) for i in range(40)]


def test_batch_equals_single(enc, settings):
    idx = build_control_index(_catalog(), enc, settings)
    batch = map_statements(STMTS, idx, enc, settings)
    assert [r.statement_id for r in batch] == [s.statement_id for s in STMTS]
    for s, b in zip(STMTS, batch):
        assert map_single(s.text, idx, enc, settings, statement_id=s.statement_id) == b


def test_batch_size_invariance(enc):
    outs = []
    for bs in (1, 7, 64):
        st = load_settings({"batch_size": bs}, env={})
        idx = build_control_index(_catalog(), enc, st)
        outs.append(map_statements(STMTS, idx, enc, st))
    assert outs[0] == outs[1] == outs[2]
    assert (7 in [c[1] for c in enc.calls]) and (64 in [c[1] for c in enc.calls])


def test_top_k_setting_and_ties(enc):
    st = load_settings({"top_k": 9}, env={})
    idx = build_control_index(_catalog(dup_text=[6, 2]), enc, st)
    r = map_single("identical control description", idx, enc, st)
    assert [c.control_id for c in r.candidates[:2]] == ["A.5.3", "A.5.7"]   # exact tie -> catalogue order
    assert r.candidates[0].similarity == r.candidates[1].similarity
    assert len(r.candidates) == 9
    with pytest.raises(ConfigError):
        map_single("x y", idx, enc, load_settings({"top_k": 10}, env={}))


def test_csv_row_order_does_not_matter(tmp_path, enc, settings):
    rows = [("A.5.10", "ten text about access"), ("A.5.2", "two text about backups"), ("A.8.1", "eight text about devices")]
    res = []
    for perm in itertools.permutations(rows):
        p = tmp_path / "c.csv"
        p.write_text("Control_ID,Control_Text\n" + "".join(f"{a},{b}\n" for a, b in perm), encoding="utf-8")
        cat = load_control_catalog(p, source_kind="synthetic")
        idx = build_control_index(cat, enc, settings)
        res.append(map_statements(STMTS[:10], idx, enc, settings))
    assert all(r == res[0] for r in res)


def test_token_profiles_for_controls_and_statements(enc, settings):
    cat = ControlCatalog(("A.1", "A.2", "A.3"), ("short text", "a very long control text " * 3, "third"), "0" * 64, "synthetic")
    idx = build_control_index(cat, enc, settings)
    assert [p.truncated for p in idx.token_profiles] == [False, True, False] and idx.truncated_control_ids == ("A.2",)
    r = map_single("one two three four five six seven eight nine ten eleven", idx, enc, settings)
    assert r.token_profile == TokenProfile(13, 12, True)


def test_input_errors(enc, settings):
    idx = build_control_index(_catalog(), enc, settings)
    assert map_statements([], idx, enc, settings) == []
    with pytest.raises(InputValidationError, match="duplicate"):
        map_statements([STMTS[0], STMTS[0]], idx, enc, settings)
    with pytest.raises(InputValidationError):
        map_single("   ", idx, enc, settings)
    with pytest.raises(InputValidationError, match="statement_id"):
        map_single("valid text", idx, enc, settings, statement_id="bad id with spaces")


def test_encoder_mismatch(enc, settings):
    from _support import TestEncoder
    idx = build_control_index(_catalog(), enc, settings)
    other = TestEncoder(dim=8)
    with pytest.raises(ConfigError, match="does not match"):
        map_statements(STMTS[:2], idx, other, settings)


def test_sbert_kind_requires_ok_full(settings):
    from _support import TestEncoder

    class Report:
        ok_full = False
        runtime_manifest_sha256 = "a" * 64

    e = TestEncoder()
    e.kind = "SBERT_PRODUCTION"
    e.report = Report()
    with pytest.raises(IntegrityError, match="full model manifest required"):
        build_control_index(_catalog(), e, settings)
    e.report = None
    with pytest.raises(IntegrityError):
        build_control_index(_catalog(), e, settings)


def test_no_text_in_logs_or_results(enc, settings, caplog):
    cat = ControlCatalog(("A.1", "A.2", "A.3"), (SECRET + " control", "other control", "third control"), "0" * 64, "synthetic")
    with caplog.at_level(logging.DEBUG):
        idx = build_control_index(cat, enc, settings)
        r = map_single(SECRET + " policy " + "word " * 20, idx, enc, settings)
    logs = "\n".join(rec.getMessage() for rec in caplog.records)
    assert SECRET not in logs and "word" not in logs
    assert SECRET not in repr(r) and not any(isinstance(v, str) and SECRET in v for v in vars(r).values())
    assert "ADHOC-0001 truncated" in logs


def test_pipeline_does_not_import_model_library():
    import subprocess
    import sys
    code = ("import sys, grc_mapper.pipeline, grc_mapper.config, grc_mapper.scoring, grc_mapper.mapping; "
            "bad=[m for m in ('sentence_transformers','transformers','torch') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    from _support import child_env
    r = subprocess.run([sys.executable, "-c", code], env=child_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
