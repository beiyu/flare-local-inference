"""Norm qualification must not initialize CUDA for help or overwrite records."""

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_help_is_independent_of_original_and_preparation_runtimes():
    code = """
import sys
from qualify.glm_text_norm_reference import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert not any(name in sys.modules for name in ('torch', 'vllm', 'b12x'))
"""
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
        capture_output=True, text=True, timeout=30)


def test_existing_actual_cuda_record_is_immutable(tmp_path):
    from qualify.glm_text_norm_reference import main

    reference = tmp_path / 'actual.safetensors'
    reference.write_bytes(b'original CUDA output')
    with pytest.raises(FileExistsError):
        main(['record', '--checkpoint', str(tmp_path), '--reference', str(reference)])
    assert reference.read_bytes() == b'original CUDA output'
