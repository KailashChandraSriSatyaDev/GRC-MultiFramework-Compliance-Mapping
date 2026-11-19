"""Environment validation for the locked research environment (generalised from the WP1 validator, revision 4).

    python tools/validate_env.py --config <LOCAL_TOML> [--prefix <ENV_PREFIX>] [--report-dir <DIR>]

Checks (one process):
  V0 interpreter version equals --python (default 3.13.11); sys.prefix equals --prefix if given
  V1 the requested packages import together and their installed versions equal the lock pins
  V2 OpenMP / MKL / BLAS inventory (Windows-specific rules; see v2_failures)
  V3 CUDA detection only, reported (no GPU compute; not a pass condition unless --require-cuda)
  V4 production model: OK_FULL verification, offline load, post-load re-verification
     (grc_mapper.pipeline.load_encoder)
Nothing is installed or changed. Paths come from arguments and the local configuration only.
The offline environment variables are set before V1 imports any ML library.
Exit code 0 only if every required check passes.

The pure V2 rule functions (ncase, dll_kind, intel_openmp_runtime_paths, v2_failures) are
importable without importing any ML library.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = REPO_ROOT / "requirements" / "grc-mvp.lock.txt"
FORBIDDEN_CONDA = ("intel-openmp", "llvm-openmp", "libiomp", "intel-cmplr-")  # plus any name starting "mkl"
IMPORT_NAMES = {"scikit-learn": "sklearn", "sentence-transformers": "sentence_transformers"}


# ------------------------------------------------------------------------------------------ pure V2 rules

def ncase(p):
    return os.path.normcase(os.path.realpath(p))


def dll_kind(fname):
    """Runtime family of a DLL from its file name. libiompstubs* are stubs, not performance runtimes."""
    f = os.path.basename(str(fname).replace("\\", "/")).lower()
    if f.startswith("libiompstubs"):
        return "intel_openmp_stub"
    if f.startswith("libiomp"):
        return "intel_openmp"
    if f.startswith("libomp"):
        return "llvm_openmp"
    if f.startswith("vcomp"):
        return "msvc_openmp"
    if f.startswith("mkl") or "_mkl" in f:
        return "mkl"
    if "openblas" in f:
        return "openblas"
    return None


def intel_openmp_runtime_paths(loaded, tp):
    """(performance_paths, stub_paths) from loaded-module entries ({'path'}) and threadpoolctl entries."""
    perf, stub = set(), set()
    for x in loaded or []:
        k = dll_kind(x.get("path", ""))
        if k == "intel_openmp":
            perf.add(ncase(x["path"]))
        elif k == "intel_openmp_stub":
            stub.add(ncase(x["path"]))
    for i in tp or []:
        if i.get("internal_api") != "openmp" or not i.get("filepath"):
            continue
        k = dll_kind(i["filepath"])
        if k == "intel_openmp":
            perf.add(ncase(i["filepath"]))
        elif k == "intel_openmp_stub":
            stub.add(ncase(i["filepath"]))
    return perf, stub


def v2_failures(conda_meta_errors, forbidden, on_disk, perf_loaded, enumeration_failed):
    """V2 decision rules. on_disk entries: {kind, file, owner}."""
    fails = []
    if conda_meta_errors:
        fails.append(f"conda-meta metadata could not be read/parsed: {conda_meta_errors}")
    if forbidden:
        fails.append(f"conda OpenMP/MKL packages installed: {forbidden}")
    iomp5md = [d for d in on_disk if os.path.basename(str(d["file"]).replace("\\", "/")).lower() == "libiomp5md.dll"]
    if len(iomp5md) > 1:
        fails.append(f"duplicate libiomp5md.dll copies: {[d['file'] for d in iomp5md]}")
    if len(perf_loaded) > 1:
        fails.append(f"more than one Intel OpenMP performance runtime loaded: {sorted(perf_loaded)}")
    mkl_not_torch = [d for d in on_disk if d["kind"] == "mkl" and not str(d["owner"]).startswith("torch wheel")]
    if mkl_not_torch:
        fails.append(f"MKL DLLs not bundled in the torch wheel: {[(d['file'], d['owner']) for d in mkl_not_torch]}")
    if enumeration_failed:
        fails.append("could not enumerate loaded modules")
    return fails


def lock_pins(lock_text):
    """{distribution: version} for the requested top-level packages named in the lock header."""
    m = re.search(r"^# Requested \(top-level\): (.+)$", lock_text, re.MULTILINE)
    if not m:
        raise ValueError("lock header lacks the requested-packages line")
    requested = [x.strip().lower() for x in m.group(1).split(",") if x.strip()]
    pins = {}
    for line in lock_text.splitlines():
        line = line.strip()
        mm = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line)
        if mm:
            pins[mm.group(1).lower()] = mm.group(2)
            continue
        mu = re.match(r"^([A-Za-z0-9_.-]+) @ \S*/([A-Za-z0-9_]+)-([^-/]+)-cp\d+", line)
        if mu:
            pins[mu.group(1).lower()] = unquote(mu.group(3))
    missing = [r for r in requested if r not in pins]
    if missing:
        raise ValueError(f"requested packages without a pin: {missing}")
    return {r: pins[r] for r in requested}


# ------------------------------------------------------------------------------------------ V2 inventory helpers

def conda_packages(prefix):
    out, errors = {}, []
    metas = sorted(glob.glob(os.path.join(prefix, "conda-meta", "*.json")))
    if not metas:
        errors.append({"file": "conda-meta", "error": "no conda-meta/*.json files found"})
    for meta in metas:
        try:
            with open(meta, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as exc:
            errors.append({"file": os.path.basename(meta), "error": repr(exc)})
            continue
        if not isinstance(d, dict) or not isinstance(d.get("name"), str) or not d["name"].strip():
            errors.append({"file": os.path.basename(meta), "error": "missing or invalid 'name'"})
            continue
        files = d.get("files", [])
        if not isinstance(files, list):
            errors.append({"file": os.path.basename(meta), "error": "'files' is not a list"})
            files = []
        out[d["name"]] = {"version": d.get("version"),
                          "files": {ncase(os.path.join(prefix, f)) for f in files if isinstance(f, str)}}
    return out, errors


def pip_file_owners():
    from importlib import metadata
    owners = {}
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "?").lower()
        for f in dist.files or []:
            try:
                owners[ncase(str(dist.locate_file(f)))] = name
            except Exception:
                pass
    return owners


def loaded_modules_windows():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModulesEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p),
                                           wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    psapi.EnumProcessModulesEx.restype = wintypes.BOOL
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD
    hproc = k32.GetCurrentProcess()
    n = 4096
    arr = (ctypes.c_void_p * n)()
    needed = wintypes.DWORD(0)
    if not psapi.EnumProcessModulesEx(hproc, arr, ctypes.sizeof(arr), ctypes.byref(needed), 0x03):
        return None
    count = min(n, needed.value // ctypes.sizeof(ctypes.c_void_p))
    paths, buf = [], ctypes.create_unicode_buffer(1024)
    for i in range(count):
        if arr[i] and psapi.GetModuleFileNameExW(hproc, arr[i], buf, 1024):
            paths.append(buf.value)
    return paths


def classify(path, pip_owner, conda_pkgs, torch_dir):
    p = ncase(path)
    if p in pip_owner:
        return "torch wheel" if pip_owner[p] == "torch" else f"pip:{pip_owner[p]}"
    for name, info in conda_pkgs.items():
        if p in info["files"]:
            return f"conda:{name}"
    if torch_dir and p.startswith(torch_dir + os.sep):
        return "torch wheel (directory; not in RECORD)"
    return "unowned"


# ------------------------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate the locked grc-mvp environment (read-only).")
    ap.add_argument("--config", default=os.environ.get("GRC_CONFIG"), help="local TOML for V4 (default GRC_CONFIG)")
    ap.add_argument("--prefix", help="expected sys.prefix of the environment")
    ap.add_argument("--python", default="3.13.11", help="expected interpreter version")
    ap.add_argument("--lock", default=str(DEFAULT_LOCK))
    ap.add_argument("--report-dir", help="directory for the JSON report (outside the repository)")
    ap.add_argument("--require-cuda", action="store_true")
    args = ap.parse_args(argv)

    # Offline variables BEFORE any ML import (V1 imports the model libraries, which read them at import).
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from grc_mapper.offline import enforce_offline_env
    enforce_offline_env()

    results = {}

    def record(key, ok, **details):
        results[key] = {"pass": bool(ok), **details}
        print(f"[{'PASS' if ok else 'FAIL'}] {key}: " + json.dumps(details, default=str)[:600])

    # V0
    want = tuple(int(x) for x in args.python.split("."))
    v0 = sys.version_info[:3] == want
    if args.prefix:
        v0 = v0 and ncase(sys.prefix) == ncase(args.prefix)
    record("V0_interpreter", v0, version=".".join(map(str, sys.version_info[:3])), prefix_checked=bool(args.prefix))

    # V1
    torch = None
    try:
        from importlib import metadata
        pins = lock_pins(Path(args.lock).read_text(encoding="utf-8"))
        imported = {}
        for dist in pins:
            mod = __import__(IMPORT_NAMES.get(dist, dist.replace("-", "_")))
            imported[dist] = mod
        torch = imported.get("torch")
        got = {d: metadata.version(d) for d in pins}
        mism = {d: {"installed": got[d], "lock": v} for d, v in pins.items() if got[d] != v}
        record("V1_imports_and_pins", not mism, installed=got, mismatches=mism)
    except Exception as exc:
        record("V1_imports_and_pins", False, error=repr(exc))

    # V2
    try:
        prefix = sys.prefix
        conda_pkgs, conda_errors = conda_packages(prefix)
        forbidden = sorted(f"{n} {i['version']}" for n, i in conda_pkgs.items()
                           if n.startswith("mkl") or any(n.startswith(x) for x in FORBIDDEN_CONDA))
        pip_owner = pip_file_owners()
        torch_dir = ncase(os.path.dirname(torch.__file__)) if torch is not None else None
        on_disk = []
        for root, _dirs, files in os.walk(prefix):
            for f in files:
                if f.lower().endswith(".dll") and dll_kind(f):
                    p = os.path.join(root, f)
                    on_disk.append({"kind": dll_kind(f), "file": os.path.relpath(p, prefix),
                                    "owner": classify(p, pip_owner, conda_pkgs, torch_dir)})
        loaded_paths = loaded_modules_windows()
        loaded = [{"kind": dll_kind(p), "path": p} for p in (loaded_paths or []) if dll_kind(p)]
        tp = []
        try:
            from threadpoolctl import threadpool_info
            tp = [{k: i.get(k) for k in ("user_api", "internal_api", "prefix", "filepath", "num_threads")}
                  for i in threadpool_info()]
        except ImportError:
            pass
        perf, stub = intel_openmp_runtime_paths(loaded, tp)
        fails = v2_failures(conda_errors, forbidden, on_disk, perf,
                            enumeration_failed=(loaded_paths is None and os.name == "nt"))
        record("V2_openmp_mkl_inventory", not fails, failures=fails, performance_runtimes=len(perf), stubs=len(stub),
               dlls_on_disk=len(on_disk), platform=sys.platform)
    except Exception as exc:
        record("V2_openmp_mkl_inventory", False, error=repr(exc), trace=traceback.format_exc()[-800:])

    # V3
    try:
        avail = bool(torch is not None and torch.cuda.is_available())
        info = {"cuda_available": avail, "torch_cuda_build": getattr(getattr(torch, "version", None), "cuda", None)}
        if avail:
            info["device_name"] = torch.cuda.get_device_name(0)
        record("V3_cuda_detection", avail or not args.require_cuda, **info)
    except Exception as exc:
        record("V3_cuda_detection", not args.require_cuda, error=repr(exc))

    # V4
    try:
        from grc_mapper.config import load_settings
        from grc_mapper.pipeline import load_encoder
        s = load_settings({"config": args.config})
        enc = load_encoder(s)
        record("V4_model_ok_full_offline", True, status=enc.report.status, files_verified=enc.report.runtime_files_verified,
               embedding_dim=enc.embedding_dim, max_seq_length=enc.max_seq_length)
    except Exception as exc:
        record("V4_model_ok_full_offline", False, error=f"{exc.__class__.__name__}: {exc}")

    ok = all(r["pass"] for r in results.values())
    report = {"tool": "validate_env", "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "all_pass": ok, "results": results}
    if args.report_dir:
        rd = Path(args.report_dir)
        try:
            rd.resolve().relative_to(REPO_ROOT.resolve())
            print("error: --report-dir must be outside the repository", file=sys.stderr)
            return 2
        except ValueError:
            pass
        rd.mkdir(parents=True, exist_ok=True)
        out = rd / f"validate_env_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"report: {out.name}")
    print(f"ALL PASS: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
