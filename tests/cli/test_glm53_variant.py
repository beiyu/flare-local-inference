"""Actual sparse safetensors headers drive EXL3 image preparation and memory plans."""

import json
import math
from pathlib import Path

import pytest

from ignition import planning
from ignition.errors import UsageError
from ignition.preparation.exl3 import tensor_contract as expert_contract
from ignition.preparation.glm53_vision import tensor_contract as vision_contract
from ignition.recipes import resolve


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("glm53-exl3-headers")
    config = dict(architectures=["Glm5NextForConditionalGeneration"], model_type="glm5_next",
        quantization_config=dict(quant_method="exl3", bits=4, codebook="mcg", scope="glm53_routed_experts_only",
            non_routed_dtype_policy="official_source_native"), text_config=dict(swiglu_limit=10.),
        vision_config=dict(hidden_size=1024, num_heads=16, depth=24, intermediate_size=4096,
            out_hidden_size=4096, projection_intermediate_size=10240, patch_size=14,
            temporal_patch_size=2, spatial_merge_size=2, in_channels=3, hidden_act="silu", rms_norm_eps=1e-5))
    (root / "config.json").write_text(json.dumps(config))
    contracts = dict(vision_contract())
    for layer in range(3, 45):
        for expert in range(288):
            contracts.update(expert_contract(layer, expert))
    contracts.update({"model.language_model.embed_tokens.weight": ("BF16", (154880, 4096)),
        "model.language_model.norm.weight": ("BF16", (4096,)), "lm_head.weight": ("BF16", (154880, 4096)),
        "model.language_model.layers.0.self_attn.q_proj.weight": ("BF16", (8192, 4096)),
        "model.language_model.layers.0.self_attn.f_a_proj.weight": ("BF16", (128, 4096)),
        "model.language_model.layers.0.hc_attn_fn": ("BF16", (4, 4096)),
        "model.language_model.layers.3.mlp.gate.weight": ("F32", (288, 4096)),
        "model.language_model.layers.45.draft.weight": ("BF16", (64, 64))})
    header, offset = {}, 0
    for name, (dtype, shape) in contracts.items():
        size = math.prod(shape) * {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4}[dtype]
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + size])
        offset += size
    raw = json.dumps(header).encode()
    # Write real headers and bounded sparse payload ranges; planning reads no tensor payloads.
    with (root / "model.safetensors").open("wb") as source:
        source.write(len(raw).to_bytes(8, "little")); source.write(raw)
        source.truncate(8 + len(raw) + offset)
    (root / "model.safetensors.index.json").write_text(json.dumps(dict(metadata=dict(total_size=offset),
        weight_map={name: "model.safetensors" for name in contracts})))
    return root, offset


def test_local_exl3_selects_image_catalog_and_exact_tp2_retained_weight_accounting(checkpoint):
    root, stored = checkpoint
    model = resolve(str(root))
    recipe, selected = model.recipe, model.checkpoint
    assert recipe.name == "glm53" and recipe.tensor_parallel == (2,)
    assert recipe.max_images == 20 and recipe.image_bytes == 2048 * 4096 * 2
    assert recipe.speculation is None and recipe.max_draft_tokens == 0
    assert selected.stored_bytes == stored and selected.draft_bytes == 0
    assert selected.expansion_bytes == 0 and selected.mapped_host_bytes == 0
    visual = sum(math.prod(shape) * 2 for _, shape in vision_contract().values())
    # Each TP2 rank holds half of each trellis, replicated gate/up input scales,
    # replicated down output scales, and half of their other scale axes.
    experts = 42 * 288 * (3 * 4096 * 2048 // 4 + (2 * (4096 + 1024) + 1024 + 4096) * 2)
    pointer_tables = 42 * 288 * 9 * 8
    non_routed = (154880 * 4096 * 2 + 4096 * 2 + 154880 * 4096
        + 8192 * 4096 + 128 * 4096 * 2 + 4 * 4096 * 4 + 288 * 4096 * 4)
    per_rank = visual + experts + pointer_tables + non_routed
    assert selected.device_bytes == 2 * per_rank
    assert recipe.weights_bytes(selected, 2, None) == per_rank
    gpus = tuple(planning.Gpu(index, "NVIDIA GB10", f"gpu{index}", 128 << 30, 120 << 30, (12, 1)) for index in range(2))
    plan = planning.fit(recipe, selected, gpus, concurrency=2, reserve_bytes=0, speculation=None, image_concurrency=1)
    assert plan.image_concurrency == 1 and plan.max_images == 20 and plan.world_size == 2
    assert plan.weights_bytes == per_rank


def test_public_nvfp4_default_remains_text_only_and_speculative():
    model = resolve("glm-5.3-flash")
    assert model.recipe.max_images == 0 and model.recipe.speculation is not None
    assert model.recipe.tensor_parallel == (1, 2, 4) and model.checkpoint.draft_bytes > 0


@pytest.mark.parametrize("field,value", [("bits", 8), ("codebook", "other"), ("scope", "all_weights")])
def test_unsupported_exl3_metadata_is_rejected_before_image_planning(checkpoint, tmp_path, field, value):
    root, _ = checkpoint
    config = json.loads((root / "config.json").read_text())
    config["quantization_config"][field] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(UsageError, match="routed-experts-only"):
        resolve(str(tmp_path))


def test_other_local_glm_precisions_keep_the_original_catalog_without_header_reads(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(dict(
        architectures=["Glm5NextForConditionalGeneration"], quantization_config=None)))
    model = resolve(str(tmp_path))
    assert model.recipe.max_images == 0 and model.recipe.speculation is not None
    assert model.recipe.tensor_parallel == (1, 2, 4)
