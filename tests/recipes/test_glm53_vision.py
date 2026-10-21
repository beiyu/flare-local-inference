"""GLM image arithmetic and serialized scratch lifetimes on real tensor execution."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from ignition.components.vision_state import PrefillVisionInputs, VisionState
from ignition.preparation.scratch import GraphBufferPool
from ignition.recipes.glm53.vision import (
    IMAGE_TOKEN, VisionScratch, clamped_swiglu, spatial_downsample, spatial_rotary,
)


def test_glm_rotary_combines_row_and_column_quarters_in_neox_order():
    source = torch.zeros((3, 1, 64), dtype=torch.float32)
    source[:, 0, [0, 16, 32, 48]] = 1
    positions = torch.tensor([[0, 0], [1, 0], [0, 1]])
    frequencies = 10000 ** (-torch.arange(0, 32, 2, dtype=torch.float32) / 32)
    angles = torch.outer(torch.arange(2, dtype=torch.float32), frequencies)
    result = spatial_rotary(source, positions, angles.cos(), angles.sin())
    torch.testing.assert_close(result[0], source[0], rtol=0, atol=0)
    cos, sin = torch.tensor(1.).cos(), torch.tensor(1.).sin()
    torch.testing.assert_close(result[1, 0, [0, 16, 32, 48]], torch.stack((cos-sin, torch.tensor(1.), sin+cos, torch.tensor(1.))))
    torch.testing.assert_close(result[2, 0, [0, 16, 32, 48]], torch.stack((torch.tensor(1.), cos-sin, torch.tensor(1.), sin+cos)))


def test_glm_effective_vision_epsilon_preserves_the_deployed_nested_config():
    import json
    from pathlib import Path
    from ignition.preparation.glm53_vision import vision_config
    raw = json.loads((Path(__file__).parents[1] / "reference/glm53-exl3-config.json").read_text())
    actual = vision_config(raw)
    assert raw["vision_config"]["rms_norm_eps"] == 1e-5
    assert actual["block_norm_eps"] == 1e-6


def test_glm_swiglu_preserves_the_deployed_bf16_silu_store_before_multiply():
    source = torch.tensor([[-20., -2., 0., 20., -30., -4., 5., 30.]], dtype=torch.bfloat16)
    actual = clamped_swiglu(source, 10.)
    gate = torch.tensor([[-20., -2., 0., 10.]])
    up = torch.tensor([[-10., -4., 5., 10.]])
    expected = (gate / (1 + (-gate).exp())).bfloat16() * up.bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A single final BF16 cast would erase the required intermediate rounding.
    samples = torch.linspace(-5, 5, 100).bfloat16()
    values = torch.stack((samples, samples.flip(0))).reshape(1, -1)
    actual = clamped_swiglu(values, 10.)
    gate, up = values.chunk(2, -1)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual != (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()) > 0
    # A lower clamp on the gate would alter this negative tail.
    assert clamped_swiglu(source, 10.)[0, 0] > 0 and clamped_swiglu(source, 10.)[0, 0] < 1e-5


def test_glm_downsample_consumes_four_spatial_patches_in_channel_first_order():
    source = torch.arange(8 * 3, dtype=torch.float32).view(8, 3)
    weights = torch.zeros((4, 3, 2, 2))
    weights[0, 0, 0, 0], weights[1, 1, 0, 1] = 1, 1
    weights[2, 2, 1, 0], weights[3, 0, 1, 1] = 1, 1
    actual = spatial_downsample(source, weights, torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(actual, torch.tensor([[0., 5., 10., 12.], [12., 17., 22., 24.]]), rtol=0, atol=0)


def test_glm_scratch_pool_keeps_every_simultaneously_live_operand_disjoint():
    pool = GraphBufferPool("cpu", chunk_bytes=1 << 20)
    scratch = VisionScratch.allocate(1024, "cpu", buffers=pool.cursor())
    nbytes = pool.nbytes
    common = (scratch.patches, scratch.residual, scratch.normalized, scratch.projected)
    stages = ((scratch.qkv, scratch.query, scratch.key, scratch.value, scratch.attended),
        (scratch.gate_up, scratch.activated),
        (scratch.merged, scratch.merge_projected, scratch.merge_gate_up, scratch.merge_activated, scratch.output))
    for stage in stages:
        live = (*common, *stage)
        intervals = sorted((tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes) for tensor in live)
        assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
        for index, tensor in enumerate(live, 1):
            tensor.fill_(index)
        for index, tensor in enumerate(live, 1):
            assert torch.all(tensor == index)
    for capacity in (256, 1024, 256):
        other = VisionScratch.allocate(capacity, "cpu", buffers=pool.cursor())
        assert pool.nbytes == nbytes
        assert all(torch.count_nonzero(getattr(other, field.name)) == 0 for field in fields(other))
    assert scratch.rows(256).output.shape == (64, 4096)
    with pytest.raises(ValueError, match="exceeds"):
        scratch.rows(4096)


def test_glm_prefill_injects_one_feature_slot_across_twenty_image_windows():
    inputs = SimpleNamespace(token_ids=torch.tensor([7, IMAGE_TOKEN, IMAGE_TOKEN, 154831]),
        row_slot_ids=torch.tensor([7]), row_generations=torch.tensor([1]))
    prefill = PrefillVisionInputs.allocate(4, "cpu")
    state = VisionState(inputs, prefill, SimpleNamespace(capacity=8), image_slots=1,
        image_token=IMAGE_TOKEN, features=2, hidden=4)
    state.feature_slot.fill_(0)
    prefill.vision_token_indices.copy_(torch.tensor([-1, 0, 1, -1]))
    for image in range(20):
        inputs.row_generations.fill_(image + 1)
        state.generation[7], state.count[7] = image + 1, 2
        state.arena[0, 0].fill_(image)
        state.arena[0, 1].fill_(image + 100)
        assert state.validate_prefill(inputs.token_ids, torch.tensor([4]))
        embedding = torch.full((4, 4), -3, dtype=torch.bfloat16)
        state.inject(inputs, embedding)
        torch.testing.assert_close(embedding, torch.tensor([[-3] * 4, [image] * 4, [image + 100] * 4, [-3] * 4], dtype=torch.bfloat16))
        state.generation[7] -= 1
        assert not state.validate_prefill(inputs.token_ids, torch.tensor([4]))
