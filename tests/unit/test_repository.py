"""Repository-level checks: lock, manifest, synthetic examples, source hygiene (U-LK-1, U-SY-1 and related)."""

import ast
import csv
import hashlib
import io
import json
import re
import tomllib

from _support import REPO_ROOT

APPROVED_LOCK = "48cdcb7762b865fcfe76a5b671ae021eddb9f38089b6ee04bae97dc31eb4483f"


def test_lock_copy_is_the_approved_lock():
    assert hashlib.sha256((REPO_ROOT / "requirements" / "grc-mvp.lock.txt").read_bytes()).hexdigest() == APPROVED_LOCK


def test_public_manifest_is_runtime_only():
    m = json.loads((REPO_ROOT / "manifests" / "model_runtime_manifest.json").read_text(encoding="utf-8"))
    assert set(m) == {"schema_version", "model_name", "description", "files"}
    paths = [f["path"] for f in m["files"]]
    assert len(paths) == 8 and "README.md" not in paths and not any(p.startswith("eval/") for p in paths)
    assert all(set(f) == {"path", "size", "sha256"} for f in m["files"])


def test_synthetic_examples():
    from grc_mapper.data import load_control_catalog, load_statements
    d = REPO_ROOT / "examples" / "synthetic"
    cat = load_control_catalog(d / "demo_controls.csv", expected_count=8, source_kind="synthetic")
    assert all(re.fullmatch(r"DEMO-\d{2}", c) for c in cat.control_ids)
    st, _ = load_statements(d / "demo_statements.csv", id_column="Statement_ID", text_column="Statement_Text",
                            expected_count=12)
    assert all(re.fullmatch(r"DS-\d{3}", s.statement_id) for s in st)


def test_pyproject_markers_and_no_install_metadata():
    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    opts = cfg["tool"]["pytest"]["ini_options"]
    assert opts["pythonpath"] == ["src"] and "-p no:cacheprovider" in opts["addopts"]
    assert {m.split(":")[0] for m in opts["markers"]} == {"unit", "integration", "regression"}
    assert "Private :: Do Not Upload" in cfg["project"]["classifiers"]


def _sources():
    for p in sorted((REPO_ROOT / "src" / "grc_mapper").glob("*.py")):
        yield p, ast.parse(p.read_text(encoding="utf-8"))


def test_no_trust_remote_code_true_and_local_files_only():
    for p, tree in _sources():
        for n in ast.walk(tree):
            if isinstance(n, ast.keyword) and n.arg == "trust_remote_code":
                assert isinstance(n.value, ast.Constant) and n.value.value is False, p.name
            if isinstance(n, ast.keyword) and n.arg == "local_files_only":
                assert isinstance(n.value, ast.Constant) and n.value.value is True, p.name


def test_no_runtime_downloads_or_installs():
    banned = ("urllib.request", "requests", "http.client", "pip", "huggingface_hub")
    for p, tree in _sources():
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                assert not any(a.name.split(".")[0] in banned or a.name in banned for a in n.names), p.name
            if isinstance(n, ast.ImportFrom) and n.module:
                assert n.module not in banned and n.module.split(".")[0] not in banned, p.name
        text = p.read_text(encoding="utf-8")
        assert "snapshot_download" not in text and "from_pretrained" not in text, p.name


def test_sentence_transformers_imported_only_in_model_loader():
    for p, tree in _sources():
        for n in ast.walk(tree):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else (
                [n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
            if any(m.split(".")[0] in ("sentence_transformers", "transformers", "torch") for m in mods):
                assert p.name == "model.py", p.name


def test_no_absolute_paths_in_source():
    pat = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][A-Za-z]|^/(home|Users|mnt|tmp)/", re.MULTILINE)
    for p, _ in _sources():
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not pat.search(node.value), (p.name, node.lineno)


def test_tier_labels_and_disclaimers_avoid_overclaiming():
    from grc_mapper import provenance
    from grc_mapper.config import TIER_DISPLAY
    assert all("confidence" not in v.lower() and "probab" not in v.lower() for v in TIER_DISPLAY.values())
    assert "not temporal drift" in provenance.DISCLAIMER and "not a compliance determination" in provenance.DISCLAIMER.lower()
    assert "not research results" in provenance.DEMO_DISCLAIMER
