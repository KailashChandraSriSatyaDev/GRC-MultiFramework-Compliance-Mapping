"""Integration tests for multi-framework mapping with the REAL production model (authoritative: grc-mvp).

Needs GRC_CONFIG and GRC_FRAMEWORKS_SPEC (local framework spec JSON). Invented statements only; outputs
go to pytest's temporary directory. Prints IDs and counts only.
"""

import csv
import io
import json
import os
import subprocess
import sys

import pytest

if not os.environ.get("GRC_CONFIG") or not os.environ.get("GRC_FRAMEWORKS_SPEC"):
    pytest.skip("GRC_CONFIG and GRC_FRAMEWORKS_SPEC are required for multi-framework integration tests",
                allow_module_level=True)

from _support import REPO_ROOT, child_env  # noqa: E402
from grc_mapper.config import load_settings  # noqa: E402
from grc_mapper.data import Statement  # noqa: E402
from grc_mapper.frameworks import load_framework_catalogue, load_framework_spec  # noqa: E402
from grc_mapper.multiframework import FrameworkIndex, map_multi  # noqa: E402
from grc_mapper.pipeline import build_control_index, load_encoder, map_statements  # noqa: E402

S = load_settings()
SPECS = load_framework_spec(os.environ["GRC_FRAMEWORKS_SPEC"])
INVENTED = [
    ("MF-001", "Administrator accounts must use multi-factor authentication and be reviewed every quarter."),
    ("MF-002", "Backups of the finance database are encrypted and restored as a test each month."),
    ("MF-003", "Visitors must sign in at reception and be escorted in secure areas."),
    ("MF-004", "Security incidents are reported to the service desk within one hour."),
    ("MF-005", "Developers must not use production data in test environments."),
]


@pytest.fixture(scope="module")
def world():
    enc = load_encoder(S)
    fws = [FrameworkIndex(sp, build_control_index(load_framework_catalogue(sp), enc, sp.scoring_settings(S)))
           for sp in SPECS]
    stmts = [Statement(i, t) for i, t in INVENTED]
    return enc, fws, stmts


def test_mf_1_catalogues_verified_and_independent(world):
    enc, fws, _ = world
    assert enc.report.status == "OK_FULL"
    for fw in fws:
        cat = fw.index.catalog
        assert len(cat.control_ids) == len(set(cat.control_ids))
        if fw.spec.expected_controls:
            assert len(cat) == fw.spec.expected_controls
        print(f"{fw.spec.framework_id} {fw.spec.framework_version}: {len(cat)} controls, "
              f"{len(fw.index.truncated_control_ids)} truncated, catalogue {cat.source_sha256[:12]}")


def test_mf_2_multi_equals_single_framework_path(world):
    enc, fws, stmts = world
    res = map_multi(stmts, fws, enc, S)
    for fw in fws:
        ref = map_statements(stmts, fw.index, enc, fw.spec.scoring_settings(S))
        assert res[fw.spec.framework_id] == ref
        print(fw.spec.framework_id, [(r.statement_id, r.top1_control_id, round(r.top1_similarity, 4), r.tier_code)
                                     for r in res[fw.spec.framework_id]])


def test_mf_3_cli_map_multi_end_to_end(tmp_path):
    cols = S.columns
    p = tmp_path / "invented.csv"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([cols["statement_id"], cols["statement_text"], "role"])
    for i, (sid, text) in enumerate(INVENTED):
        w.writerow([sid, text, "external_test" if i < 4 else "excluded"])
    p.write_text(buf.getvalue(), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "grc_mapper", "map-multi", "--statements", str(p), "--frameworks",
                        os.environ["GRC_FRAMEWORKS_SPEC"], "--out", str(tmp_path / "runs"), "--select-column", "role",
                        "--select-value", "external_test"], env=child_env(), capture_output=True, cwd=str(REPO_ROOT),
                       timeout=900)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-2000:]
    out = json.loads(r.stdout)
    run = tmp_path / "runs" / out["run_id"]
    prov = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    assert prov["model"]["verification_status"] == "OK_FULL" and prov["inputs"]["rows_selected"] == 4
    assert [f["framework_id"] for f in prov["frameworks"]] == [sp.framework_id for sp in SPECS]
    for sp in SPECS:
        rows = list(csv.DictReader(io.StringIO((run / f"results_{sp.framework_id}.csv").read_text(encoding="utf-8"))))
        assert len(rows) == 4 and {x["framework_version"] for x in rows} == {sp.framework_version}
    for f in run.iterdir():
        blob = f.read_text(encoding="utf-8")
        assert not any(t in blob for _, t in INVENTED)
