"""Fail-closed publication scanner for the public repository (Blueprint rev 3, section 6.3).

Run manually before any commit (no hook is installed):

    python tools/check_publishable.py --config <LOCAL_TOML> --final

Coverage: the ENTIRE public root, walked recursively on the filesystem (os.scandir, links never
followed). Git is never called and .gitignore has no effect, so hidden, ignored and untracked
files are all scanned. The only structural exception is the root `.git/` directory (Git's
compressed object store), which is counted but not content-scanned; a --final PASS is therefore
required immediately before every commit.

Checks (each PASS, FAIL or NOT_RUN):
  S1 every file is on the allowlist; no symlinks, junctions or reparse points
  S2 size <= 1 MB; text is UTF-8 without NUL bytes; no blocked extension
  S3 generic personal-path and e-mail patterns
  S4 local identity tokens (from the local config; NOT_RUN without them)
  S5 8-word shingle overlap with the local control and policy texts (NOT_RUN without them)
  S6 restricted hashes (NOT_RUN without the local registry); approved runtime-model hashes are
     allowed only in manifests/model_runtime_manifest.json and docs/REPRODUCIBILITY.md
  S7 examples/ free of research-claim wording
  S8 the lock copy matches the approved SHA-256
Overall: FAIL (exit 1) if any check failed; otherwise INCOMPLETE (exit 2) if any check did not
run; otherwise PASS (exit 0). --final accepts only PASS. It exits 2 before scanning when no
configuration is given (neither --config nor a non-empty GRC_CONFIG), when the named file does not
exist, or when the configuration or an input it names cannot be read.

The report lists file, line and check ID only. Matched text is never printed or saved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APPROVED_LOCK_SHA256 = "48cdcb7762b865fcfe76a5b671ae021eddb9f38089b6ee04bae97dc31eb4483f"
LOCK_PATH = "requirements/grc-mvp.lock.txt"
APPROVED_HASH_LOCATIONS = frozenset({"manifests/model_runtime_manifest.json", "docs/REPRODUCIBILITY.md"})
MAX_BYTES = 1 << 20
SHINGLE = 8
CHECKS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
_FILE_ATTRIBUTE_HIDDEN = 0x2
_FILE_ATTRIBUTE_SYSTEM = 0x4
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400

ALLOWLIST = tuple(re.compile(p) for p in (
    r"^README\.md$",
    r"^\.gitignore$",
    r"^\.gitattributes$",
    r"^pyproject\.toml$",
    r"^requirements/grc-mvp\.lock\.txt$",
    r"^config/example\.local\.toml$",
    r"^manifests/model_runtime_manifest\.json$",
    r"^src/grc_mapper/[A-Za-z0-9_]+\.py$",
    r"^examples/synthetic/(README\.md|demo_controls\.csv|demo_statements\.csv)$",
    r"^tests/([A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py$",
    r"^tools/(validate_env|check_publishable)\.py$",
    r"^docs/[A-Za-z0-9_]+\.md$",
))
BLOCKED_EXTENSIONS = frozenset({
    ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".ckpt", ".pkl", ".npy", ".npz", ".ipynb",
    ".xlsx", ".xls", ".parquet", ".jsonl", ".tsv", ".zip", ".7z", ".tar", ".gz", ".log", ".pyc", ".db",
})
# Built from pieces so this file never matches its own patterns.
_SEP = "\\\\"
GENERIC_PATTERNS = tuple(re.compile(p) for p in (
    "[A-Za-z]:" + _SEP + "Users" + _SEP,          # Windows user profile path
    "/" + "Users" + "/[^/\\s]+/",                   # macOS user path
    "/" + "home" + "/[^/\\s]+/",                    # Linux home path
    "[A-Za-z0-9._%+-]+" + "@" + "[A-Za-z0-9-]+(\\.[A-Za-z0-9-]+)*\\.[A-Za-z]{2,}",  # e-mail address
))
CLAIM_WORDS = re.compile(r"\b(" + "|".join(("track", "accuracy", "validated")) + r")\b", re.IGNORECASE)
HEX64 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
WORD = re.compile(r"[a-z0-9]+")


@dataclass
class CheckResult:
    status: str = "PASS"  # PASS | FAIL | NOT_RUN
    reason: str | None = None
    findings: list[dict] = field(default_factory=list)

    def fail(self, path: str, line: int | None = None, detail: str | None = None) -> None:
        self.status = "FAIL"
        f = {"path": path, "line": line}
        if detail:
            f["detail"] = detail
        self.findings.append(f)

    def not_run(self, reason: str) -> None:
        if self.status != "FAIL":
            self.status = "NOT_RUN"
        self.reason = reason if self.reason is None else self.reason


@dataclass
class Entry:
    rel: str
    path: Path
    size: int
    hidden: bool


# ------------------------------------------------------------------------------------------ walk

def _attrs(st: os.stat_result) -> int:
    return int(getattr(st, "st_file_attributes", 0) or 0)


def walk(root: Path, results: dict[str, CheckResult]) -> tuple[list[Entry], dict]:
    """Recursive filesystem walk (no Git, links never followed)."""
    files: list[Entry] = []
    meta = {"git_dir_present": False, "git_files": 0, "git_bytes": 0, "empty_dirs": [], "hidden_files": 0,
            "unreadable": []}

    def visit(d: Path, rel: str) -> None:
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as exc:
            meta["unreadable"].append(rel or ".")
            for c in CHECKS[:7]:
                results[c].not_run(f"directory could not be listed ({exc.__class__.__name__})")
            return
        if not entries and rel:
            meta["empty_dirs"].append(rel)
        for e in entries:
            r = f"{rel}/{e.name}" if rel else e.name
            try:
                st = e.stat(follow_symlinks=False)
            except OSError:
                meta["unreadable"].append(r)
                for c in CHECKS[:7]:
                    results[c].not_run("an entry could not be inspected")
                continue
            is_link = e.is_symlink() or bool(_attrs(st) & _FILE_ATTRIBUTE_REPARSE_POINT)
            if not is_link and hasattr(e, "is_junction") and e.is_junction():
                is_link = True
            if is_link:
                results["S1"].fail(r, None, "symlink, junction or reparse point")
                continue
            if stat.S_ISDIR(st.st_mode):
                if r == ".git":
                    meta["git_dir_present"] = True
                    _count_git(Path(e.path), meta)
                    continue
                visit(Path(e.path), r)
            elif stat.S_ISREG(st.st_mode):
                hidden = e.name.startswith(".") or bool(_attrs(st) & (_FILE_ATTRIBUTE_HIDDEN | _FILE_ATTRIBUTE_SYSTEM))
                meta["hidden_files"] += int(hidden)
                files.append(Entry(r, Path(e.path), st.st_size, hidden))
            else:
                results["S1"].fail(r, None, "non-regular file")

    visit(root, "")
    return files, meta


def _count_git(d: Path, meta: dict) -> None:
    for dirpath, _dirs, names in os.walk(d, followlinks=False):
        for n in names:
            meta["git_files"] += 1
            try:
                meta["git_bytes"] += os.lstat(os.path.join(dirpath, n)).st_size
            except OSError:
                pass


# ------------------------------------------------------------------------------------------ local inputs

def load_local_inputs(config: Path | None) -> dict:
    """Identity tokens, shingle source texts and the restricted-hash registry from the local config."""
    out: dict = {"identity_tokens": None, "texts": None, "blocked": None, "approved": None,
                 "scan_output_dir": None, "errors": []}
    if config is None:
        return out
    import tomllib
    try:
        doc = tomllib.loads(Path(config).read_text(encoding="utf-8-sig"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        out["errors"].append(f"config unreadable ({exc.__class__.__name__})")
        return out
    paths = doc.get("paths", {})
    scanner = doc.get("scanner", {})
    cols = doc.get("columns", {})
    tokens = scanner.get("identity_tokens")
    if isinstance(tokens, list) and tokens and all(isinstance(t, str) and t.strip() for t in tokens):
        out["identity_tokens"] = [t.casefold() for t in tokens]
    if scanner.get("scan_output_dir"):
        out["scan_output_dir"] = Path(os.path.expanduser(os.path.expandvars(scanner["scan_output_dir"])))
    texts: list[str] = []
    ok = True
    for key, col in (("controls_csv", cols.get("control_text", "Control_Text")),
                     ("policies_csv", cols.get("statement_text", "Policy_Text"))):
        p = paths.get(key)
        if not p:
            ok = False
            continue
        try:
            raw = Path(os.path.expanduser(os.path.expandvars(p))).read_bytes().decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(raw, newline="")))
            if not rows or col not in rows[0]:
                ok = False
                continue
            texts.extend(r[col] for r in rows if r.get(col))
        except (OSError, UnicodeDecodeError, csv.Error):
            ok = False
    if ok and texts:
        out["texts"] = texts
    reg = paths.get("restricted_hashes")
    if reg:
        try:
            d = json.loads(Path(os.path.expanduser(os.path.expandvars(reg))).read_text(encoding="utf-8"))
            out["blocked"] = {e["sha256"].lower() for e in d["blocked"]}
            out["approved"] = {h.lower() for h in d["approved_runtime_hashes"]}
        except (OSError, ValueError, KeyError, TypeError):
            out["errors"].append("restricted-hash registry unreadable")
    return out


def shingles(text: str) -> set[tuple[str, ...]]:
    w = WORD.findall(text.lower())
    return {tuple(w[i:i + SHINGLE]) for i in range(len(w) - SHINGLE + 1)}


# ------------------------------------------------------------------------------------------ scan

def scan(root: Path, local: dict) -> dict:
    results = {c: CheckResult() for c in CHECKS}
    files, meta = walk(root, results)

    if local.get("identity_tokens") is None:
        results["S4"].not_run("no identity tokens configured")
    shingle_set: set = set()
    if local.get("texts") is None:
        results["S5"].not_run("local control/policy CSVs not configured or unreadable")
    else:
        for t in local["texts"]:
            shingle_set |= shingles(t)
    if local.get("blocked") is None:
        results["S6"].not_run("restricted-hash registry not configured or unreadable")
    blocked = local.get("blocked") or set()
    approved = local.get("approved") or set()

    lock_seen = False
    for f in files:
        rel = f.rel
        if not any(p.match(rel) for p in ALLOWLIST):
            results["S1"].fail(rel, None, "not on the allowlist")
        if Path(rel).suffix.lower() in BLOCKED_EXTENSIONS:
            results["S2"].fail(rel, None, "blocked extension")
        if f.size > MAX_BYTES:
            results["S2"].fail(rel, None, "larger than 1 MB")
            continue
        try:
            data = f.path.read_bytes()
        except OSError:
            for c in ("S2", "S3", "S4", "S5", "S6", "S7"):
                results[c].not_run("a file could not be read")
            continue
        digest = hashlib.sha256(data).hexdigest()
        if rel == LOCK_PATH:
            lock_seen = True
            if digest != APPROVED_LOCK_SHA256:
                results["S8"].fail(rel, None, "lock differs from the approved lock")
        if blocked and digest in blocked and digest not in approved:
            results["S6"].fail(rel, None, "file is byte-identical to a restricted asset")
        if b"\x00" in data:
            results["S2"].fail(rel, None, "binary content")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            results["S2"].fail(rel, None, "not valid UTF-8")
            continue
        lines = text.splitlines()
        for n, line in enumerate(lines, start=1):
            for pat in GENERIC_PATTERNS:
                if pat.search(line):
                    results["S3"].fail(rel, n)
                    break
            if local.get("identity_tokens"):
                low = line.casefold()
                if any(tok in low for tok in local["identity_tokens"]):
                    results["S4"].fail(rel, n)
            for m in HEX64.finditer(line):
                h = m.group(0).lower()
                if h in approved:
                    if rel not in APPROVED_HASH_LOCATIONS:
                        results["S6"].fail(rel, n, "approved runtime hash outside its allowed files")
                elif h in blocked:
                    results["S6"].fail(rel, n, "restricted hash")
            if rel.startswith("examples/") and CLAIM_WORDS.search(line):
                results["S7"].fail(rel, n)
        if shingle_set:
            hit_line = _first_shingle_line(lines, shingle_set)
            if hit_line is not None:
                results["S5"].fail(rel, hit_line)
    if not lock_seen:
        results["S8"].fail(LOCK_PATH, None, "lock copy missing")

    statuses = [r.status for r in results.values()]
    overall = "FAIL" if "FAIL" in statuses else ("INCOMPLETE" if "NOT_RUN" in statuses else "PASS")
    return {
        "schema_version": 1,
        "overall": overall,
        "checks": {c: {"status": r.status, "reason": r.reason, "findings": r.findings} for c, r in results.items()},
        "files_scanned": len(files),
        "hidden_files_scanned": meta["hidden_files"],
        "empty_directories": meta["empty_dirs"],
        "unreadable_entries": len(meta["unreadable"]),
        "git_metadata": {"present": meta["git_dir_present"], "files": meta["git_files"], "bytes": meta["git_bytes"],
                         "content_scanned": False},
        "local_input_errors": local.get("errors", []),
        "scanned_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _first_shingle_line(lines: list[str], shingle_set: set) -> int | None:
    words: list[tuple[str, int]] = []
    for n, line in enumerate(lines, start=1):
        words.extend((w, n) for w in WORD.findall(line.lower()))
    for i in range(len(words) - SHINGLE + 1):
        if tuple(w for w, _ in words[i:i + SHINGLE]) in shingle_set:
            return words[i][1]
    return None


# ------------------------------------------------------------------------------------------ CLI

def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fail-closed publication scanner (filesystem walk; never uses Git).")
    ap.add_argument("--root", default=str(REPO_ROOT), help="public repository root (default: this repository)")
    ap.add_argument("--config", default=os.environ.get("GRC_CONFIG"), help="local TOML (default: GRC_CONFIG)")
    ap.add_argument("--report-dir", help="where to save scan_report.json (must be outside the root)")
    ap.add_argument("--final", action="store_true", help="pre-commit mode: only PASS is accepted")
    args = ap.parse_args(argv)
    root = Path(args.root)
    config = args.config.strip() if isinstance(args.config, str) and args.config.strip() else None
    if args.final and config is None:
        print("error: --final requires the local configuration (--config or GRC_CONFIG)", file=sys.stderr)
        return 2
    if config is not None and not Path(config).is_file():
        print("error: configuration file not found", file=sys.stderr)
        return 2
    local = load_local_inputs(Path(config) if config else None)
    if args.final and local.get("errors"):
        print("error: --final: the local configuration or an input it names could not be read: "
              + "; ".join(local["errors"]), file=sys.stderr)
        return 2
    report = scan(root, local)

    report_dir = Path(args.report_dir) if args.report_dir else local.get("scan_output_dir")
    saved = None
    if report_dir is not None:
        if _inside(report_dir, root):
            print("error: the report directory must be outside the scanned root", file=sys.stderr)
            return 2
        out = Path(report_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(3))
        out.mkdir(parents=True, exist_ok=False)
        saved = out / "scan_report.json"
        saved.write_bytes((json.dumps(report, indent=2, ensure_ascii=True) + "\n").encode("ascii"))
    elif args.final:
        report["overall"] = "INCOMPLETE" if report["overall"] == "PASS" else report["overall"]
        report["report_not_saved"] = True

    for c in CHECKS:
        r = report["checks"][c]
        n = len(r["findings"])
        extra = f" ({r['reason']})" if r["reason"] else ""
        print(f"{c} {r['status']}: {n} finding(s){extra}")
        for f in r["findings"][:200]:
            loc = f"{f['path']}:{f['line']}" if f["line"] else f["path"]
            print(f"    {c} {loc}" + (f"  [{f['detail']}]" if f.get("detail") else ""))
    print(f"files scanned: {report['files_scanned']} (hidden: {report['hidden_files_scanned']}); "
          f".git present: {report['git_metadata']['present']} (not content-scanned)")
    if saved is not None:
        print(f"report saved: {saved.name} in {saved.parent.name}")
    print(f"OVERALL: {report['overall']}" + (" (final mode)" if args.final else ""))
    return {"PASS": 0, "FAIL": 1}.get(report["overall"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
