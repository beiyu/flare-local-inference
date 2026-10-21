"""CPU range-copy tests for preserving the deployed transplant and template."""
import hashlib
import json
from pathlib import Path

import pytest
from qualify.glm_migration_checkpoint import create, plan, LAYERS, DONOR_REVISION


def shard(path, tensors):
    payload = b""; header = {}
    for name, value in tensors.items():
        header[name] = {"dtype": "BF16", "shape": [len(value) // 2], "data_offsets": [len(payload), len(payload) + len(value)]}
        payload += value
    raw = json.dumps(header).encode(); path.write_bytes(len(raw).to_bytes(8, "little") + raw + payload)


@pytest.fixture
def source(tmp_path):
    checkpoint = tmp_path / "source"; checkpoint.mkdir()
    donors = tmp_path / "donors"; donors.mkdir()
    tensors = {}; entries = {}
    for layer in LAYERS:
        key = f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
        tensors[key] = b"orig"; value = bytes([layer]) * 4
        (donors / f"L{layer}.bin").write_bytes(value)
        entries[str(layer)] = {"key": key, "dtype": "BF16", "shape": [2], "nbytes": 4,
                               "sha256": hashlib.sha256(value).hexdigest()}
    tensors["unrelated"] = b"keep"
    shard(checkpoint / "patched.safetensors", tensors)
    shard(checkpoint / "unchanged.safetensors", {"other": b"same"})
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 4 * (len(tensors) + 1)},
        "weight_map": {**{key: "patched.safetensors" for key in tensors}, "other": "unchanged.safetensors"}}))
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["Glm5NextForConditionalGeneration"],
        "quantization_config": {"quant_method": "exl3"}}))
    (checkpoint / "tokenizer_config.json").write_text(json.dumps({"chat_template": "old"}))
    (donors / "MANIFEST.json").write_text(json.dumps({"donor_sha": DONOR_REVISION, "layers": entries}))
    template = tmp_path / "deployed.jinja"; template.write_text("deployed tools and thinking template")
    return checkpoint, donors, template, tmp_path / "output"


def test_transplant_keeps_sources_ranges_and_template_independent(source):
    checkpoint, donors, template, output = source
    before = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    manifest, patches, _, copied = plan(checkpoint, donors, template)
    assert copied == (checkpoint / "patched.safetensors").stat().st_size
    assert create(*source) == output
    assert (output / "unchanged.safetensors").is_symlink()
    assert not (output / "patched.safetensors").is_symlink()
    assert {path.name: path.read_bytes() for path in checkpoint.iterdir()} == before
    assert (output / "chat_template.jinja").read_bytes() == template.read_bytes()
    assert json.loads((output / "tokenizer_config.json").read_text())["chat_template"] == template.read_text()
    tensor = manifest.tensors["unrelated"]
    raw = (output / tensor.shard).read_bytes()
    assert raw[tensor.offset:tensor.offset + tensor.nbytes] == b"keep"
    for tensor, donor, _ in patches:
        assert raw[tensor.offset:tensor.offset + tensor.nbytes] == donor.read_bytes()
    with pytest.raises(ValueError, match="already exist"): create(*source)


@pytest.mark.parametrize("failure", ["hash", "shape", "revision", "missing"])
def test_invalid_donor_never_creates_output(source, failure):
    checkpoint, donors, template, output = source
    description = json.loads((donors / "MANIFEST.json").read_text())
    if failure == "hash": (donors / "L15.bin").write_bytes(b"oops")
    elif failure == "shape": description["layers"]["15"]["shape"] = [1, 2]
    elif failure == "revision": description["donor_sha"] = "different"
    else: del description["layers"]["45"]
    (donors / "MANIFEST.json").write_text(json.dumps(description))
    with pytest.raises(ValueError): create(*source)
    assert not output.exists()
