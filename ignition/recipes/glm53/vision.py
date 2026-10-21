"""Prepared BF16 GLM image encoder with spatial rotary attention and patch merger."""

from dataclasses import dataclass

from ignition.components.glm_vision_kernels import original_rms_out, original_add_rms, original_qk

import torch
import torch.nn.functional as F

PATCH_CAPACITIES = (64, 256, 1024, 4096, 8192)
FEATURES = 2048
HIDDEN = 4096
IMAGE_TOKEN = 154854
PIXEL_WIDTH = 1176


def clamped_swiglu(source, limit):
    gate, up = source.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    # The deployed CUDA activation stores SiLU in the input dtype before the
    # final multiplication, including when the checkpoint uses BF16.
    activated = F.silu(gate.float()).to(source.dtype)
    return activated * up.clamp(-limit, limit)


def spatial_rotary(source, positions, cos_cache, sin_cache):
    cos = cos_cache[positions.long()].flatten(1)[:, None, :].float()
    sin = sin_cache[positions.long()].flatten(1)[:, None, :].float()
    left, right = source.float().chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), -1).to(source.dtype)


def spatial_downsample(source, weight, bias):
    merged = source.view(-1, 2, 2, source.shape[-1]).permute(0, 3, 1, 2)
    return F.conv2d(merged, weight, bias, stride=2).flatten(1)


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_num_features: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)
        n = PATCH_CAPACITIES[-1]
        return cls(zeros((n, PIXEL_WIDTH), torch.float32), zeros((n, 2), torch.int64), zeros(2),
                   zeros(1), zeros(1), zeros(1) - 1, zeros(1, torch.int64), zeros(1))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    residual: torch.Tensor
    normalized: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    projected: torch.Tensor
    gate_up: torch.Tensor
    activated: torch.Tensor
    merged: torch.Tensor
    merge_projected: torch.Tensor
    merge_gate_up: torch.Tensor
    merge_activated: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, device, *, buffers=None):
        if patches not in PATCH_CAPACITIES:
            raise ValueError("GLM vision scratch must use a declared patch capacity")
        def zeros(cursor, rows, width):
            return (torch.zeros((rows, width), dtype=torch.bfloat16, device=device) if cursor is None
                    else cursor.zeros((rows, width), torch.bfloat16))
        pixels = zeros(buffers, patches, PIXEL_WIDTH)
        residual, normalized, projected = (zeros(buffers, patches, 1024) for _ in range(3))
        stages = [buffers.fork() for _ in range(3)] if buffers is not None else [None] * 3
        attention, mlp, merger = stages
        qkv = zeros(attention, patches, 3072)
        query, key, value = (zeros(attention, patches, 1024).view(patches, 16, 64) for _ in range(3))
        attended = zeros(attention, patches, 1024)
        gate_up, activated = zeros(mlp, patches, 8192), zeros(mlp, patches, 4096)
        merged, merge_projected = (zeros(merger, patches // 4, 4096) for _ in range(2))
        merge_gate_up = zeros(merger, patches // 4, 20480)
        merge_activated = zeros(merger, patches // 4, 10240)
        output = zeros(merger, patches // 4, HIDDEN)
        if buffers is not None:
            buffers.join(*stages)
        return cls(pixels, residual, normalized, qkv, query, key, value, attended, projected,
                   gate_up, activated, merged, merge_projected, merge_gate_up, merge_activated, output)

    def rows(self, patches):
        if patches not in PATCH_CAPACITIES or patches > self.patches.shape[0]:
            raise ValueError("GLM vision view exceeds its prepared capacity")
        merged = {"merged", "merge_projected", "merge_gate_up", "merge_activated", "output"}
        return type(self)(*(getattr(self, name)[:patches // 4 if name in merged else patches]
                            for name in self.__dataclass_fields__))


class PreparedVision:
    """All Python here executes during preparation and capture; Helios replays the graph."""

    def __init__(self, weights, scratch, inputs, *, kernels, session, workspace, name, buffers=None):
        from ignition.components.linear import UnquantizedProjection
        from ignition.components.varlen_attention import PreparedVarlenAttention

        self.weights, self.scratch, self.inputs = weights, scratch, inputs
        self.kernels = kernels
        if weights.config["block_norm_eps"] != 1e-6:
            raise ValueError("the deployed GLM vision RMSNorm image requires epsilon 1e-6")
        n, device = scratch.patches.shape[0], scratch.patches.device
        def zeros(shape, dtype=torch.int32):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        self.positions = zeros((n, 2), torch.int64)
        self.cu, self.num_patches, self.num_features = zeros((2,)), zeros((1,)), zeros((1,))
        frequencies = 1. / (10000 ** (torch.arange(0, 32, 2, dtype=torch.float32, device=device) / 32))
        angles = torch.outer(torch.arange(8192, dtype=torch.float32, device=device), frequencies)
        # The deployed get_rope caches cos/sin in BF16 before rotating FP32 values.
        self.cos, self.sin = angles.cos().bfloat16(), angles.sin().bfloat16()
        def projection(source, weight, target, suffix, bias=None):
            return UnquantizedProjection(source, weight, target, session=session, name=name + suffix, bias=bias)
        s = scratch
        self.patch = projection(s.patches, weights.patch, s.residual, ".patch", weights.patch_bias)
        self.blocks = []
        attention = PreparedVarlenAttention(s.query, s.key, s.value, self.cu,
            s.attended.view_as(s.query), session=session, workspace=workspace, name=name + ".attention")
        for index, w in enumerate(weights.layers):
            self.blocks.append((w,
                projection(s.normalized, w.qkv, s.qkv, f".{index}.qkv", w.qkv_bias),
                projection(s.attended, w.output, s.projected, f".{index}.output", w.output_bias),
                projection(s.normalized, w.gate_up, s.gate_up, f".{index}.gate_up", w.gate_up_bias),
                projection(s.activated, w.down, s.projected, f".{index}.down", w.down_bias)))
        self.attention = attention
        self.merge_project = projection(s.merged, weights.merge_project, s.merge_projected, ".merger.proj")
        self.merge_up = projection(s.merge_projected, weights.merge_gate_up, s.merge_gate_up, ".merger.gate_up")
        self.merge_down = projection(s.merge_activated, weights.merge_down, s.output, ".merger.down")
        self.output_rows = torch.arange(n // 4, device=device)

    def __call__(self):
        s, w = self.scratch, self.weights
        self.patch(s.patches)
        for layer, qkv, output, up, down in self.blocks:
            original_rms_out(s.residual, layer.norm1, s.normalized, self.kernels.handle)
            q, k, v = qkv(s.normalized).view(-1, 3, 16, 64).unbind(1)
            s.query.copy_(q)
            s.key.copy_(k)
            original_qk(s.query, s.key, layer.q_norm, layer.k_norm, self.kernels.handle)
            s.query.copy_(spatial_rotary(s.query, self.positions, self.cos, self.sin))
            s.key.copy_(spatial_rotary(s.key, self.positions, self.cos, self.sin))
            s.value.copy_(v)
            self.attention()
            attended = output(s.attended)
            s.normalized.copy_(s.residual)
            original_add_rms(s.normalized, attended, layer.norm2, self.kernels.handle)
            s.residual.copy_(attended)
            s.activated.copy_(clamped_swiglu(up(s.normalized), w.config["swiglu_limit"]))
            s.residual.add_(down(s.activated))
        original_rms_out(s.residual, w.norm, s.normalized, self.kernels.handle)
        s.merged.copy_(spatial_downsample(s.normalized, w.downsample, w.downsample_bias))
        projected = self.merge_project(s.merged)
        s.merge_projected.copy_(F.gelu(F.layer_norm(projected, (HIDDEN,), w.merge_norm, w.merge_norm_bias, 1e-5)))
        s.merge_activated.copy_(clamped_swiglu(self.merge_up(s.merge_projected), w.config["swiglu_limit"]))
        output = self.merge_down(s.merge_activated)
        output.masked_fill_((self.output_rows >= self.num_features)[:, None], 0)
        return output
