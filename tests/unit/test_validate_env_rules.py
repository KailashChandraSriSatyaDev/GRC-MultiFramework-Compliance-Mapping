"""Unit tests for the pure V2 OpenMP rules and lock parsing in tools/validate_env.py (ported WP1 rule tests).

Paths are neutral examples; no ML library is imported.
"""

import importlib.util
import sys

import pytest

from _support import REPO_ROOT

SPEC = importlib.util.spec_from_file_location("validate_env", REPO_ROOT / "tools" / "validate_env.py")
ve = importlib.util.module_from_spec(SPEC)
sys.modules["validate_env"] = ve
SPEC.loader.exec_module(ve)

T = "X:\\envs\\grc-mvp\\Lib\\site-packages\\torch\\lib"
PERF = T + "\\libiomp5md.dll"
STUB = T + "\\libiompstubs5md.dll"
OTHER_PERF = "X:\\envs\\grc-mvp\\Library\\bin\\libiomp5md.dll"
TORCH_IOMP = "Lib\\site-packages\\torch\\lib\\libiomp5md.dll"


def disk(*files):
    return [{"kind": ve.dll_kind(f), "file": f, "owner": "torch wheel"} for f in files]


def tp_entry(path, threads):
    return {"user_api": "openmp", "internal_api": "openmp", "prefix": "libiomp", "filepath": path, "num_threads": threads}


def test_no_ml_imports():
    assert not [m for m in ("torch", "numpy", "sentence_transformers") if m in sys.modules and m == "torch"]


@pytest.mark.parametrize("name,kind", [
    ("libiomp5md.dll", "intel_openmp"), ("libiompstubs5md.dll", "intel_openmp_stub"), (STUB, "intel_openmp_stub"),
    ("libiomp5.dll", "intel_openmp"), ("vcomp140.dll", "msvc_openmp"), ("libscipy_openblas64_.dll", "openblas"),
    ("mkl_core.2.dll", "mkl"), ("libomp140.x86_64.dll", "llvm_openmp"), ("kernel32.dll", None),
])
def test_dll_kind(name, kind):
    assert ve.dll_kind(name) == kind


def test_runtime_alone_passes():
    perf, stub = ve.intel_openmp_runtime_paths([{"path": PERF}], [tp_entry(PERF, 8)])
    assert len(perf) == 1 and not stub and not ve.v2_failures([], [], disk(TORCH_IOMP), perf, False)


def test_runtime_plus_stub_passes():
    perf, stub = ve.intel_openmp_runtime_paths([{"path": PERF}, {"path": STUB}], [tp_entry(PERF, 8), tp_entry(STUB, 1)])
    f = ve.v2_failures([], [], disk(TORCH_IOMP, "Lib\\site-packages\\torch\\lib\\libiompstubs5md.dll"), perf, False)
    assert len(perf) == 1 and len(stub) == 1 and not f


def test_two_performance_runtimes_fail():
    perf, _ = ve.intel_openmp_runtime_paths([{"path": PERF}, {"path": OTHER_PERF}], [tp_entry(PERF, 8)])
    assert any("performance runtime" in x for x in ve.v2_failures([], [], disk(TORCH_IOMP), perf, False))
    perf, _ = ve.intel_openmp_runtime_paths([{"path": PERF}], [tp_entry(PERF, 8), tp_entry(OTHER_PERF, 8)])
    assert any("performance runtime" in x for x in ve.v2_failures([], [], disk(TORCH_IOMP), perf, False))


def test_duplicate_on_disk_fails():
    perf, _ = ve.intel_openmp_runtime_paths([{"path": PERF}], [tp_entry(PERF, 8)])
    f = ve.v2_failures([], [], disk(TORCH_IOMP, "Library\\bin\\libiomp5md.dll"), perf, False)
    assert any("duplicate libiomp5md.dll" in x for x in f)


def test_neighbouring_rules():
    assert any("conda-meta" in x for x in ve.v2_failures([{"file": "x.json", "error": "bad"}], [], [], set(), False))
    assert any("conda OpenMP/MKL" in x for x in ve.v2_failures([], ["intel-openmp 2025.1"], [], set(), False))
    conda_mkl = [{"kind": "mkl", "file": "Library\\bin\\mkl_core.2.dll", "owner": "conda:mkl"}]
    assert any("MKL DLLs not bundled" in x for x in ve.v2_failures([], [], conda_mkl, set(), False))
    torch_mkl = [{"kind": "mkl", "file": "Lib\\site-packages\\torch\\lib\\mkl_x.dll", "owner": "torch wheel"}]
    assert not ve.v2_failures([], [], torch_mkl, set(), False)
    assert any("enumerate" in x for x in ve.v2_failures([], [], [], set(), True))


def test_lock_pins_from_approved_lock():
    pins = ve.lock_pins((REPO_ROOT / "requirements" / "grc-mvp.lock.txt").read_text(encoding="utf-8"))
    assert pins == {"torch": "2.10.0+cu130", "numpy": "2.4.1", "openpyxl": "3.1.5", "pandas": "2.3.3",
                    "pytest": "9.2.0", "scikit-learn": "1.7.1", "scipy": "1.16.3",
                    "sentence-transformers": "5.3.0", "transformers": "5.3.0"}


def test_lock_pins_rejects_incomplete():
    with pytest.raises(ValueError):
        ve.lock_pins("# Requested (top-level): numpy, torch\nnumpy==1.0 \\\n")
    with pytest.raises(ValueError):
        ve.lock_pins("numpy==1.0\n")
