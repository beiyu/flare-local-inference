"""EXL3 preparation validates real safetensors geometry before CUDA loading."""

import json
import math

import pytest

from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.exl3 import (
    row_slices, tensor_contract, tensor_extent, validate_config, validate_tensors,
)


def checkpoint(tmp_path, change=None):
    header, offset = {}, 0
    for name, (dtype, shape) in tensor_contract(3, 287).items():
        if change is not None:
            dtype, shape = change(name, dtype, shape)
        size = math.prod(shape) * {"F16": 2, "I16": 2, "I32": 4}[dtype]
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + size])
        offset += size
    raw = json.dumps(header).encode()
    with (tmp_path / "model.safetensors").open("wb") as source:
        source.write(len(raw).to_bytes(8, "little"))
        source.write(raw)
        source.truncate(8 + len(raw) + offset)
    return inspect_indexed_checkpoint(tmp_path)


def test_full_checkpoint_contract_and_target_layer_boundaries(tmp_path):
    manifest = checkpoint(tmp_path)
    assert validate_tensors(manifest, 3, [287]) == (287,)
    for ids in [[], [287, 287], [True], [288]]:
        with pytest.raises(ValueError):
            validate_tensors(manifest, 3, ids)
    for layer in [2, 45, True]:
        with pytest.raises(ValueError, match="target expert layer"):
            tensor_contract(layer, 287)
    with pytest.raises(ValueError, match="geometry or dtype"):
        validate_tensors(manifest, 3, [0])


@pytest.mark.parametrize("suffix", ["gate_proj.trellis", "up_proj.svh", "down_proj.suh", "down_proj.mcg"])
def test_wrong_packed_geometry_is_rejected_before_device_transfer(tmp_path, suffix):
    def change(name, dtype, shape):
        return (dtype, (1, math.prod(shape))) if name.endswith(suffix) else (dtype, shape)
    if suffix.endswith("mcg"):
        def change(name, dtype, shape):
            return ("I16", shape) if name.endswith(suffix) else (dtype, shape)
    with pytest.raises(ValueError, match="geometry or dtype"):
        validate_tensors(checkpoint(tmp_path, change), 3, [287])


def test_tp2_slices_partition_original_payload_axes_without_requantization():
    for name, (_, shape) in tensor_contract(3, 287).items():
        projection, suffix = name.rsplit(".", 2)[-2:]
        extents = [tensor_extent(projection, suffix, rank) for rank in (0, 1)]
        if extents == [None, None]:
            assert suffix == "mcg" or (projection != "down_proj" and suffix == "suh") or (projection == "down_proj" and suffix == "svh")
            continue
        axes = [extent if isinstance(extent, tuple) else (extent,) for extent in extents]
        split = [axis for axis, pair in enumerate(zip(*axes)) if pair[0].stop is not None]
        assert len(split) == 1
        axis = split[0]
        first, second = axes[0][axis], axes[1][axis]
        assert (first.start, first.stop, second.start, second.stop) == (0, shape[axis] // 2, shape[axis] // 2, shape[axis])


def test_explicit_native_non_routed_policy_and_tp2_are_required():
    config = dict(model_type="glm5_next", architectures=["Glm5NextForConditionalGeneration"],
        quantization_config=dict(quant_method="exl3", bits=4, codebook="mcg",
            scope="glm53_routed_experts_only", non_routed_dtype_policy="official_source_native"))
    validate_config(config, 2)
    for key, value in [("bits", 3), ("scope", "all"), ("codebook", "mul1"),
                       ("non_routed_dtype_policy", "mxfp8")]:
        changed = dict(config, quantization_config=dict(config["quantization_config"], **{key: value}))
        with pytest.raises(ValueError, match="native non-routed"):
            validate_config(changed, 2)
    for world in [1, 4, True]:
        with pytest.raises(ValueError, match="TP2"):
            validate_config(config, world)


@pytest.mark.parametrize("rows", [1, 31, 32, 33, 65, 4096])
def test_static_slices_cover_the_whole_prefill_and_decode_catalog(rows):
    slices = row_slices(rows)
    assert [row for first, last in slices for row in range(first, last)] == list(range(rows))
    assert all(0 < last - first <= 32 for first, last in slices)


@pytest.mark.parametrize("rows", [0, -1, True, 1.5])
def test_invalid_row_catalog_is_rejected(rows):
    with pytest.raises(ValueError, match="positive integer"):
        row_slices(rows)


def test_original_nested_linear_attention_metadata_and_legacy_flat_schema():
    from pathlib import Path
    from ignition.recipes.glm53.model import validate_linear_attention_config

    config = json.loads(Path(__file__).with_name("glm53-exl3-config.json").read_text())
    validate_config(config, 2)
    text = config["text_config"]
    validate_linear_attention_config(text)
    legacy = dict(text)
    nested = legacy.pop("linear_attn_config")
    for flat, key in (("linear_num_heads", "num_heads"), ("linear_head_dim", "head_dim"),
                      ("linear_conv_kernel_dim", "short_conv_kernel_size")):
        legacy[flat] = nested[key]
    validate_linear_attention_config(legacy)
    for malformed in [dict(text, linear_num_heads=32), dict(legacy, linear_conv_kernel_dim=8),
                      dict(text, linear_attn_config=None)]:
        with pytest.raises(ValueError, match="GLM"):
            validate_linear_attention_config(malformed)
