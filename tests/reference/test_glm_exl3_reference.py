"""Reference-tool boundaries must fail before loading a CUDA/reference runtime."""

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_help_does_not_initialize_reference_or_preparation_runtime():
    script = """
import sys
from qualify.glm_exl3_reference import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert not any(name in sys.modules for name in ('torch', 'vllm', 'b12x', 'flare_heliopause'))
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True,
        capture_output=True, text=True, timeout=30)


def test_existing_reference_is_never_overwritten(tmp_path):
    from qualify.glm_exl3_reference import main

    reference = tmp_path / "original.safetensors"
    reference.write_bytes(b"immutable reference")
    with pytest.raises(FileExistsError):
        main(["record", "--checkpoint", str(tmp_path), "--reference", str(reference), "--tp-rank", "0"])
    assert reference.read_bytes() == b"immutable reference"


@pytest.mark.parametrize("bounds", [[], ["--rtol", "0"], ["--rtol", "nan", "--atol", "0"],
    ["--rtol", "0", "--atol", "-1"], ["--rtol", "inf", "--atol", "0"]])
def test_check_requires_explicit_finite_arithmetic_bounds_before_cuda(tmp_path, bounds):
    result = subprocess.run([sys.executable, "-m", "qualify.glm_exl3_reference", "check",
        "--checkpoint", str(tmp_path), "--reference", str(tmp_path / "missing.safetensors"),
        "--library", str(tmp_path / "missing.so"), "--prefill-library", str(tmp_path / "missing-e3.so"), *bounds], cwd=ROOT,
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "set finite nonnegative" in result.stderr
    assert "No module named" not in result.stderr


def test_unvalidated_prefill_extension_is_rejected_without_loading_torch_or_cuda(tmp_path):
    from ignition.preparation.exl3_prefill import extract_images

    wrong = tmp_path / "wrong.so"
    wrong.write_bytes(b"different extension")
    with pytest.raises(ValueError, match="deployed thin/E3 binary"):
        extract_images(wrong)
