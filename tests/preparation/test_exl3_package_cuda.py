"""Actual EXL3 weights and deployed CUDA binary through a fresh Rust process."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned idle SM121 GPU")
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def native_checker():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES explicitly"
    for name in ("FLARE_EXL3_CHECKPOINT", "FLARE_EXL3_LIBRARY"):
        assert os.environ.get(name), f"set {name} to the actual EXL3 probe input"
    subprocess.run(["cargo", "build", "--manifest-path", str(ROOT / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin",
        "test_fixed_runtime_package"], cwd=ROOT, check=True, timeout=180)
    return ROOT / "helios/target/debug/test_fixed_runtime_package"


@pytest.mark.parametrize("rows", [1, 32])
def test_exl3_capture_exports_and_reloads_without_python(tmp_path, native_checker, rows):
    result = subprocess.run([sys.executable, "-m", "qualify.exl3_package", "run",
        "--checkpoint", os.environ["FLARE_EXL3_CHECKPOINT"],
        "--library", os.environ["FLARE_EXL3_LIBRARY"], "--output", str(tmp_path / "probe"),
        "--rows", str(rows), "--native-executable", str(native_checker)], cwd=ROOT,
        check=True, capture_output=True, text=True, timeout=300)
    assert "native fixed-address replay passed" in result.stdout
    assert "full GLM remains unqualified" in result.stdout
