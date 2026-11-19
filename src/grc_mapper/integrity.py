"""SHA-256 helpers and strict model-directory verification (Blueprint rev 3, sections 4.2 and 4.5).

`verify_model_dir()` uses only the standard library and never imports sentence_transformers,
so it always completes before the model library is imported or the model is loaded.

Verification statuses:
- OK_FULL: every runtime file matches the public runtime manifest, every present expected
  non-runtime file matches the local full manifest, the directory layout is allowed and no
  other file exists. Required before every production-model load.
- OK_PARTIAL: as above, but no full manifest was supplied, so present non-runtime files are
  `present_unverified`. Diagnostic only; it never permits a model load.
Anything else raises IntegrityError.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .errors import IntegrityError

#: The closed list of expected non-runtime paths. Exact paths, not patterns.
NON_RUNTIME_PATHS: frozenset[str] = frozenset(
    {"README.md", "eval/similarity_evaluation_train_dev_results.csv"}
)
#: Allowed directories and their required state.
ALLOWED_DIRECTORIES: dict[str, str] = {"1_Pooling": "any", "eval": "any", "2_Normalize": "empty"}
SCHEMA_VERSION = 1
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400

VerificationStatus = Literal["OK_FULL", "OK_PARTIAL"]
NonRuntimeState = Literal["verified", "absent", "present_unverified"]


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class RuntimeManifest:
    model_name: str
    files: tuple[ManifestEntry, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class FullManifest:
    model_name: str
    runtime_files: tuple[ManifestEntry, ...]
    non_runtime_files: tuple[ManifestEntry, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class FileStamp:
    path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class ManifestReport:
    status: VerificationStatus
    runtime_files_verified: int
    non_runtime: tuple[tuple[str, NonRuntimeState], ...]
    directories: tuple[str, ...]
    runtime_manifest_sha256: str
    full_manifest_sha256: str | None
    stamps: tuple[FileStamp, ...]

    @property
    def ok_full(self) -> bool:
        return self.status == "OK_FULL"


# --------------------------------------------------------------------------- hashing

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (lower-case hex)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- manifests

def _check_rel_path(p: object, where: str) -> str:
    if not isinstance(p, str) or not p:
        raise IntegrityError(f"{where}: path must be a non-empty string")
    if "\\" in p or p.startswith("/") or re.match(r"^[A-Za-z]:", p):
        raise IntegrityError(f"{where}: path {p!r} must be relative and use '/' separators")
    parts = p.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise IntegrityError(f"{where}: path {p!r} contains an empty, '.' or '..' segment")
    return p


def _parse_entries(raw: object, where: str) -> tuple[ManifestEntry, ...]:
    if not isinstance(raw, list) or not raw:
        raise IntegrityError(f"{where}: expected a non-empty list of file entries")
    out: list[ManifestEntry] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        loc = f"{where}[{i}]"
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise IntegrityError(f"{loc}: entry must have exactly the keys path, size, sha256")
        path = _check_rel_path(item["path"], loc)
        size = item["size"]
        digest = item["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise IntegrityError(f"{loc}: size must be a non-negative integer")
        if not isinstance(digest, str) or not _SHA_RE.match(digest):
            raise IntegrityError(f"{loc}: sha256 must be 64 lower-case hex characters")
        if path in seen:
            raise IntegrityError(f"{loc}: duplicate path {path!r}")
        seen.add(path)
        out.append(ManifestEntry(path, size, digest))
    return tuple(sorted(out, key=lambda e: e.path))


def _read_json(path: Path, what: str) -> tuple[dict, str]:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise IntegrityError(f"{what} cannot be read: {Path(path).name} ({exc.__class__.__name__})") from None
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise IntegrityError(f"{what} is not valid UTF-8 JSON: {Path(path).name}") from None
    if not isinstance(obj, dict):
        raise IntegrityError(f"{what} must be a JSON object: {Path(path).name}")
    if obj.get("schema_version") != SCHEMA_VERSION:
        raise IntegrityError(f"{what}: unsupported schema_version in {Path(path).name}")
    return obj, sha256_bytes(data)


def load_runtime_manifest(path: Path) -> RuntimeManifest:
    """Load the public runtime manifest (runtime files only)."""
    obj, digest = _read_json(path, "runtime manifest")
    files = _parse_entries(obj.get("files"), "runtime manifest files")
    bad = sorted(e.path for e in files if e.path in NON_RUNTIME_PATHS)
    if bad:
        raise IntegrityError(f"runtime manifest lists non-runtime paths: {bad}")
    name = obj.get("model_name")
    if not isinstance(name, str) or not name:
        raise IntegrityError("runtime manifest: model_name missing")
    return RuntimeManifest(name, files, digest)


def load_full_manifest(path: Path | None) -> FullManifest | None:
    """Load the local full manifest (runtime + expected non-runtime files), or None."""
    if path is None:
        return None
    obj, digest = _read_json(path, "full manifest")
    runtime = _parse_entries(obj.get("runtime_files"), "full manifest runtime_files")
    non_runtime = _parse_entries(obj.get("non_runtime_files"), "full manifest non_runtime_files")
    extra = sorted(e.path for e in non_runtime if e.path not in NON_RUNTIME_PATHS)
    if extra:
        raise IntegrityError(f"full manifest lists non-runtime paths outside the closed list: {extra}")
    overlap = {e.path for e in runtime} & {e.path for e in non_runtime}
    if overlap:
        raise IntegrityError(f"full manifest lists paths as both runtime and non-runtime: {sorted(overlap)}")
    name = obj.get("model_name")
    if not isinstance(name, str) or not name:
        raise IntegrityError("full manifest: model_name missing")
    return FullManifest(name, runtime, non_runtime, digest)


# --------------------------------------------------------------------------- directory walk

def _is_link_or_reparse(entry: os.DirEntry) -> bool:
    if entry.is_symlink():
        return True
    is_junction = getattr(entry, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    try:
        st = entry.stat(follow_symlinks=False)
    except OSError:
        return True
    return bool(getattr(st, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _walk(root: Path) -> tuple[dict[str, os.stat_result], dict[str, int], list[str]]:
    """Return (files, directory child counts, problems). Never follows links."""
    files: dict[str, os.stat_result] = {}
    dirs: dict[str, int] = {}
    problems: list[str] = []

    def visit(path: Path, rel: str) -> int:
        count = 0
        try:
            with os.scandir(path) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as exc:
            problems.append(f"cannot list {rel or '.'} ({exc.__class__.__name__})")
            return 0
        for entry in entries:
            count += 1
            child = f"{rel}/{entry.name}" if rel else entry.name
            if _is_link_or_reparse(entry):
                problems.append(f"symlink, junction or reparse point not allowed: {child}")
                continue
            st = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(st.st_mode):
                dirs[child] = visit(Path(entry.path), child)
            elif stat.S_ISREG(st.st_mode):
                files[child] = st
            else:
                problems.append(f"non-regular file not allowed: {child}")
        return count

    visit(root, "")
    return files, dirs, problems


# --------------------------------------------------------------------------- verification

def verify_model_dir(
    model_dir: Path,
    runtime: RuntimeManifest,
    full: FullManifest | None,
) -> ManifestReport:
    """Strictly verify a model directory. Raises IntegrityError on any failure."""
    root = Path(model_dir)
    try:
        root_st = os.lstat(root)
    except OSError:
        raise IntegrityError(f"model directory not found: {root.name}") from None
    if stat.S_ISLNK(root_st.st_mode) or getattr(root_st, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise IntegrityError("model directory must not be a symlink, junction or reparse point")
    if not stat.S_ISDIR(root_st.st_mode):
        raise IntegrityError(f"model path is not a directory: {root.name}")

    if full is not None:
        if full.model_name != runtime.model_name:
            raise IntegrityError("full manifest model_name differs from the runtime manifest")
        if full.runtime_files != runtime.files:
            raise IntegrityError("full manifest runtime_files differ from the runtime manifest")

    files, dirs, problems = _walk(root)

    for d, n_children in sorted(dirs.items()):
        if d not in ALLOWED_DIRECTORIES:
            problems.append(f"unexpected directory: {d}/")
        elif ALLOWED_DIRECTORIES[d] == "empty" and n_children:
            problems.append(f"directory must be empty: {d}/")

    runtime_paths = {e.path for e in runtime.files}
    for rel in sorted(files):
        if rel not in runtime_paths and rel not in NON_RUNTIME_PATHS:
            problems.append(f"unexpected file: {rel}")

    verified = 0
    for e in runtime.files:
        st = files.get(e.path)
        if st is None:
            problems.append(f"runtime file missing: {e.path}")
            continue
        if st.st_size != e.size:
            problems.append(f"runtime file size mismatch: {e.path}")
            continue
        if sha256_file(root / e.path) != e.sha256:
            problems.append(f"runtime file sha256 mismatch: {e.path}")
            continue
        verified += 1

    expected_nr = {e.path: e for e in full.non_runtime_files} if full is not None else {}
    nr_states: list[tuple[str, NonRuntimeState]] = []
    for p in sorted(NON_RUNTIME_PATHS):
        st = files.get(p)
        if st is None:
            nr_states.append((p, "absent"))
            continue
        if full is None:
            nr_states.append((p, "present_unverified"))
            continue
        e = expected_nr.get(p)
        if e is None:
            problems.append(f"non-runtime file present but not in the full manifest: {p}")
        elif st.st_size != e.size:
            problems.append(f"non-runtime file size mismatch: {p}")
        elif sha256_file(root / p) != e.sha256:
            problems.append(f"non-runtime file sha256 mismatch: {p}")
        else:
            nr_states.append((p, "verified"))

    if problems:
        raise IntegrityError("model directory verification failed:\n  - " + "\n  - ".join(problems))

    status: VerificationStatus = "OK_FULL" if full is not None else "OK_PARTIAL"
    stamps = tuple(FileStamp(p, st.st_size, st.st_mtime_ns) for p, st in sorted(files.items()))
    return ManifestReport(
        status=status,
        runtime_files_verified=verified,
        non_runtime=tuple(nr_states),
        directories=tuple(sorted(dirs)),
        runtime_manifest_sha256=runtime.manifest_sha256,
        full_manifest_sha256=full.manifest_sha256 if full is not None else None,
        stamps=stamps,
    )


def require_full_verification(report: ManifestReport) -> None:
    """Raise IntegrityError unless the report is OK_FULL. Nothing relaxes this check."""
    if report.status != "OK_FULL":
        raise IntegrityError("full model manifest required for production-model inference")


def reverify_after_load(
    model_dir: Path,
    runtime: RuntimeManifest,
    full: FullManifest,
    before: ManifestReport,
) -> ManifestReport:
    """Post-load re-verification against the full local manifest.

    Re-runs `verify_model_dir` in full: the closed layout check plus a fresh SHA-256 of every
    runtime file and every present expected non-runtime file. The result must be OK_FULL, and
    the size and modification time of every file must equal those recorded before the load.
    Raises IntegrityError otherwise.
    """
    if full is None:
        raise IntegrityError("full model manifest required for post-load re-verification")
    try:
        after = verify_model_dir(model_dir, runtime, full)
    except IntegrityError as exc:
        raise IntegrityError(f"model directory changed during load: {exc}") from None
    require_full_verification(after)
    if after.stamps != before.stamps:
        changed = sorted(
            {s.path for s in after.stamps} ^ {s.path for s in before.stamps}
            | {a.path for a, b in zip(after.stamps, before.stamps) if a != b}
        )
        raise IntegrityError(f"model directory changed during load (size or modification time): {changed}")
    return after
