"""EXL3 package-probe preflight reads real headers without CUDA initialization."""

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from qualify.exl3_package import checkpoint_plan, tensor_contract, validate_library


ROOT = Path(__file__).resolve().parents[2]


def checkpoint(tmp_path, *, change=None):
    config = {"architectures": ["Glm5NextForConditionalGeneration"], "quantization_config": {
        "quant_method": "exl3", "bits": 4, "codebook": "mcg", "scope": "glm53_routed_experts_only"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    header, offset = {}, 0
    for name, (dtype, shape) in tensor_contract(3, 7).items():
        if change is not None:
            dtype, shape = change(name, dtype, shape)
        size = math.prod(shape) * {"I16": 2, "F16": 2, "I32": 4}[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    # Real safetensors byte ranges; sparse payload keeps CPU fixtures bounded.
    with (tmp_path / "model.safetensors").open("wb") as source:
        source.write(len(raw).to_bytes(8, "little"))
        source.write(raw)
        source.truncate(8 + len(raw) + offset)
    return config


def test_selected_expert_header_contract_is_checked_before_weight_reads(tmp_path):
    checkpoint(tmp_path)
    manifest, contract = checkpoint_plan(tmp_path, 3, 7)
    assert len(contract) == 12
    assert all(manifest.tensors[name].dtype == dtype and manifest.tensors[name].shape == shape
               for name, (dtype, shape) in contract.items())
    with pytest.raises(ValueError, match="expected checkpoint"):
        checkpoint_plan(tmp_path, 3, 8)


def test_wrong_full_expert_geometry_is_rejected(tmp_path):
    def incorrect(name, dtype, shape):
        return dtype, (128, 128, 64) if name.endswith("gate_proj.trellis") else shape

    checkpoint(tmp_path, change=incorrect)
    with pytest.raises(ValueError, match="expected checkpoint"):
        checkpoint_plan(tmp_path, 3, 7)


def test_other_exl3_codebooks_are_not_labeled_as_validated_mcg(tmp_path):
    config = checkpoint(tmp_path)
    config["quantization_config"]["codebook"] = "mul1"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="EXL3 K4 MCG"):
        checkpoint_plan(tmp_path, 3, 7)


def test_unvalidated_library_is_rejected_without_loading_it(tmp_path):
    library = tmp_path / "different.so"
    library.write_bytes(b"not an executable library")
    with pytest.raises(ValueError, match="validated GLM"):
        validate_library(library)


def test_help_does_not_import_torch_or_initialize_cuda():
    command = """
import sys
from qualify.exl3_package import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert not any(name in sys.modules for name in ('torch', 'b12x', 'flare_heliopause'))
"""
    subprocess.run([sys.executable, "-c", command], cwd=ROOT, check=True, timeout=30,
                   capture_output=True, text=True)


@pytest.mark.parametrize("arguments,error", [
    (["--layer", "2"], "routed target layer"),
    (["--expert", "288"], "expert must be"),
    (["--rows", "33"], "invalid choice"),
])
def test_invalid_probe_selection_fails_before_output_or_cuda(tmp_path, arguments, error):
    output = tmp_path / "unused"
    result = subprocess.run([sys.executable, "-m", "qualify.exl3_package", "export",
        "--checkpoint", str(tmp_path / "missing"), "--library", str(tmp_path / "missing.so"),
        "--output", str(output), *arguments], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and error in result.stderr
    assert not output.exists()
