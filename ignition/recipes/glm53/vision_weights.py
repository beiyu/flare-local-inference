"""Checkpoint-native BF16 weights for the GLM image encoder and projector."""

from dataclasses import dataclass
import json

import torch

from ignition.components.weights import DirectWeightLoader


@dataclass(frozen=True)
class VisionLayerWeights:
    norm1: torch.Tensor
    norm2: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    qkv: torch.Tensor
    qkv_bias: torch.Tensor
    output: torch.Tensor
    output_bias: torch.Tensor
    gate_up: torch.Tensor
    gate_up_bias: torch.Tensor
    down: torch.Tensor
    down_bias: torch.Tensor


@dataclass(frozen=True)
class VisionWeights:
    config: dict
    patch: torch.Tensor
    patch_bias: torch.Tensor
    layers: tuple[VisionLayerWeights, ...]
    norm: torch.Tensor
    downsample: torch.Tensor
    downsample_bias: torch.Tensor
    merge_project: torch.Tensor
    merge_norm: torch.Tensor
    merge_norm_bias: torch.Tensor
    merge_gate_up: torch.Tensor
    merge_down: torch.Tensor

    @property
    def allocated_bytes(self):
        from dataclasses import fields
        tensors = [getattr(self, field.name) for field in fields(self)]
        tensors.extend(getattr(layer, field.name) for layer in self.layers for field in fields(layer))
        return sum(value.nbytes for value in tensors if isinstance(value, torch.Tensor))

    @classmethod
    def load(cls, manifest, device, *, progress=None):
        raw = json.loads((manifest.metadata_path / "config.json").read_text())
        from ignition.preparation.glm53_vision import vision_config, validate_tensors
        c = vision_config(raw)
        validate_tensors(manifest)
        with DirectWeightLoader(manifest, device, prefixes=("model.visual.",), progress=progress,
                                progress_label="Load BF16 image encoder") as loader:
            def tensor(name, shape):
                value = loader.load_tensor("model.visual." + name)
                if value.dtype != torch.bfloat16 or tuple(value.shape) != shape:
                    raise ValueError(f"GLM visual.{name} requires BF16 {shape}, got {value.dtype} {tuple(value.shape)}")
                return value

            def joined(prefix, suffix, shape):
                return torch.cat([tensor(prefix + name + suffix, shape) for name in ("gate_proj", "up_proj")])

            layers = []
            for i in range(c["depth"]):
                p = f"blocks.{i}."
                layers.append(VisionLayerWeights(tensor(p + "norm1.weight", (1024,)),
                    tensor(p + "norm2.weight", (1024,)), tensor(p + "attn.q_norm.weight", (64,)),
                    tensor(p + "attn.k_norm.weight", (64,)), tensor(p + "attn.qkv.weight", (3072, 1024)),
                    tensor(p + "attn.qkv.bias", (3072,)), tensor(p + "attn.proj.weight", (1024, 1024)),
                    tensor(p + "attn.proj.bias", (1024,)), joined(p + "mlp.", ".weight", (4096, 1024)),
                    joined(p + "mlp.", ".bias", (4096,)), tensor(p + "mlp.down_proj.weight", (1024, 4096)),
                    tensor(p + "mlp.down_proj.bias", (1024,))))
            return cls(c, tensor("patch_embed.proj.weight", (1024, 3, 2, 14, 14)).flatten(1),
                tensor("patch_embed.proj.bias", (1024,)), tuple(layers), tensor("post_layernorm.weight", (1024,)),
                tensor("downsample.weight", (4096, 1024, 2, 2)), tensor("downsample.bias", (4096,)),
                tensor("merger.proj.weight", (4096, 4096)), tensor("merger.post_projection_norm.weight", (4096,)),
                tensor("merger.post_projection_norm.bias", (4096,)),
                joined("merger.", ".weight", (10240, 4096)), tensor("merger.down_proj.weight", (4096, 10240)))
