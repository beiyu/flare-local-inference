"""The deployed GLM DFlash2 checkpoint, preserving BF16 weights and TP slices."""

from dataclasses import dataclass
import json
from pathlib import Path


DRAFT_DIRECTORY = "dflash2"
REPOSITORY = "incoai/GLM-5.3-Flash-DFlash2"
REVISION = "bf582e4eacc1810f76656d1811693ff6c6737d2a"


@dataclass(frozen=True)
class DraftConfig:
    block_size: int
    window: int
    target_layers: tuple[int, ...]
    mask_token: int
    group_size: int
    taps: int
    selector_rank: int
    selector_top_k: int
    epsilon: float
    rope_theta: float

    @classmethod
    def load(cls, snapshot):
        raw = json.loads((Path(snapshot) / "config.json").read_text())
        expected = dict(architectures=["DFlash2DraftModel"], hidden_size=4096, intermediate_size=12288,
            num_hidden_layers=5, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
            num_target_layers=45, vocab_size=154880, hidden_act="silu", attention_bias=False,
            is_causal=False, tie_word_embeddings=False, dtype="bfloat16",
            layer_types=["sliding_attention"] * 5)
        for key, value in expected.items():
            if raw.get(key) != value:
                raise ValueError(f"GLM DFlash2 requires {key}={value!r}")
        if raw.get("quantization_config") is not None:
            raise ValueError("GLM DFlash2 requires its original BF16 weights")
        d = raw["dflash_config"]
        geometry = dict(block_size=8, conv_group_size=16, conv_kernel_size=2,
            mask_token_id=154856, selector_rank=256, selector_top_k=16,
            target_layer_ids=[5, 14, 24, 33, 42])
        if any(d.get(key) != value for key, value in geometry.items()):
            raise ValueError("unsupported GLM DFlash2 block, feature, convolution, or selector geometry")
        if set(d) != set(geometry):
            raise ValueError("unsupported GLM DFlash2 configuration fields")
        if (raw.get("sliding_window") != 2048 or raw.get("rms_norm_eps") != 1e-5
                or raw.get("rope_parameters") != dict(rope_theta=10000.0, rope_type="default")):
            raise ValueError("unsupported GLM DFlash2 window, normalization, or rotary geometry")
        return cls(8, 2048, tuple(d["target_layer_ids"]), d["mask_token_id"], 16, 2, 256, 16, 1e-5, 10000.)

    def regions(self, world_size):
        if world_size not in (1, 2, 4):
            raise ValueError("GLM DFlash2 requires TP1, TP2, or TP4")
        return tuple((f"dflash2.{i}.{kind}", (self.window, 8 // world_size, 128), "BF16")
                     for i in range(5) for kind in ("key", "value"))


def inspect_draft(snapshot):
    """Reject incompatible metadata and all missing/extra tensor headers before GPU loading."""
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    config = DraftConfig.load(snapshot)
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    shapes = {"fc.weight": (4096, 5 * 4096), "hidden_norm.weight": (4096,), "norm.weight": (4096,),
        "candidate_selector.hidden_projection.weight": (256, 4096),
        "candidate_selector.predecessor_codebook": (154880, 256),
        "candidate_selector.successor_codebook": (154880, 256)}
    for i in range(5):
        p = f"layers.{i}."
        shapes.update({p + k: v for k, v in {
            "input_layernorm.weight": (4096,), "post_attention_layernorm.weight": (4096,),
            "self_attn.q_proj.weight": (4096, 4096), "self_attn.k_proj.weight": (1024, 4096),
            "self_attn.v_proj.weight": (1024, 4096), "self_attn.o_proj.weight": (4096, 4096),
            "self_attn.q_norm.weight": (128,), "self_attn.k_norm.weight": (128,),
            "mlp.gate_proj.weight": (12288, 4096), "mlp.up_proj.weight": (12288, 4096),
            "mlp.down_proj.weight": (4096, 12288),
            **{f"{kind}.{name}": shape for kind in ("attention_conv", "mlp_conv")
               for name, shape in (("base_kernel", (2, 2, 4096)), ("kernel_projection.weight", (1024, 4096)))},
        }.items()})
    if manifest.tensors.keys() != shapes.keys():
        raise ValueError("GLM DFlash2 checkpoint tensor names differ from the deployed 81-tensor model")
    for name, shape in shapes.items():
        tensor = manifest.tensors[name]
        if tensor.dtype != "BF16" or tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: GLM DFlash2 requires BF16 {shape}")
    return config, manifest


@dataclass(frozen=True)
class DraftWeights:
    config: DraftConfig
    projection: object
    hidden_norm: object
    norm: object
    context_kv: object
    layers: tuple[dict, ...]
    selector: dict

    @classmethod
    def load(cls, snapshot, device, shard, *, progress=None):
        import torch
        from ignition.components.weights import DirectWeightLoader, MLPWeights

        config, manifest = inspect_draft(snapshot)
        with DirectWeightLoader(manifest, device, progress=progress, progress_label="Load DFlash2 weights") as loader:
            projection, hidden_norm, norm = loader.load_tensors(("fc.weight", "hidden_norm.weight", "norm.weight"))
            layers = []
            for i in range(5):
                p = f"layers.{i}."
                qkv = loader.load_bf16_linear_group(tuple(p + "self_attn." + name for name in ("q_proj", "k_proj", "v_proj")),
                    row_ranges=(shard.partition(4096), shard.partition(1024), shard.partition(1024)))
                layer = dict(qkv=qkv, context=qkv[4096 // shard.world_size:],
                    output=loader.load_bf16_linear_group((p + "self_attn.o_proj",), column_range=shard.partition(4096)),
                    mlp=MLPWeights(loader.load_bf16_linear_group(tuple(p + "mlp." + name for name in ("gate_proj", "up_proj")),
                        row_ranges=(shard.partition(12288),) * 2),
                        loader.load_bf16_linear_group((p + "mlp.down_proj",), column_range=shard.partition(12288))),
                    **dict(zip(("input_norm", "post_norm", "q_norm", "k_norm"), loader.load_tensors(tuple(p + name for name in
                        ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"))), strict=True)),
                    **{kind: dict(zip(("base", "projection"), loader.load_tensors((p + kind + ".base_kernel",
                        p + kind + ".kernel_projection.weight")), strict=True)) for kind in ("attention_conv", "mlp_conv")})
                layers.append(layer)
            selector = dict(zip(("hidden", "predecessor", "successor"), loader.load_tensors(tuple("candidate_selector." + name
                for name in ("hidden_projection.weight", "predecessor_codebook", "successor_codebook"))), strict=True))
        return cls(config, projection, hidden_norm, norm, torch.cat([layer["context"] for layer in layers]), tuple(layers), selector)
