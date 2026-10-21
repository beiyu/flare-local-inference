"""Checkpoint contracts for the deployed routed-experts-only EXL3 representation.

These checks run without Torch or CUDA. No conversion of the trellis payload,
Hadamard scales, or non-routed weights is permitted by this representation.
"""

BITS = 4
MCG_MARKER = -877912083
HIDDEN = 4096
INTERMEDIATE = 2048
EXPERTS = 288
TOPK = 8
MAX_ROWS = 32
GEOMETRY = 1
LIBRARY_SHA256 = "9b7d4e1fc81402313d82f286f027a067bc2b5b7b0684d066096112585c06b717"
POINTER_KEYS = tuple(f"{projection}.{suffix}"
    for projection in ("gate_proj", "up_proj", "down_proj")
    for suffix in ("trellis", "suh", "svh"))


def validate_config(config, world_size):
    quant = config.get("quantization_config", {})
    if (config.get("model_type") != "glm5_next"
            or config.get("architectures") != ["Glm5NextForConditionalGeneration"]
            or quant.get("quant_method") != "exl3" or quant.get("bits") != BITS
            or quant.get("codebook") != "mcg"
            or quant.get("scope") != "glm53_routed_experts_only"
            or quant.get("non_routed_dtype_policy") != "official_source_native"):
        raise ValueError("GLM EXL3 requires routed-experts-only K4 MCG with native non-routed weights")
    if type(world_size) is not int or world_size != 2:
        raise ValueError("the prepared EXL3 cooperative geometry requires TP2")


def tensor_contract(layer, expert):
    if type(layer) is not int or not 3 <= layer < 45:
        raise ValueError("EXL3 target expert layer must be in [3, 45)")
    if type(expert) is not int or not 0 <= expert < EXPERTS:
        raise ValueError("EXL3 expert ID must be in [0, 288)")
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
    result = {}
    for projection, inputs, outputs in (("gate_proj", HIDDEN, INTERMEDIATE),
                                       ("up_proj", HIDDEN, INTERMEDIATE),
                                       ("down_proj", INTERMEDIATE, HIDDEN)):
        for suffix, dtype, shape in (("trellis", "I16", (inputs // 16, outputs // 16, BITS * 16)),
                                    ("suh", "F16", (inputs,)), ("svh", "F16", (outputs,)),
                                    ("mcg", "I32", (1,))):
            result[f"{prefix}.{projection}.{suffix}"] = dtype, shape
    return result


def tensor_extent(projection, suffix, rank):
    """Exact TP2 slices used by the deployment, before contiguous device transfer."""
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("EXL3 TP rank must be 0 or 1")
    if projection not in ("gate_proj", "up_proj", "down_proj") or suffix not in ("trellis", "suh", "svh", "mcg"):
        raise ValueError("unknown EXL3 tensor component")
    first, last = rank * 1024, (rank + 1) * 1024
    if projection in ("gate_proj", "up_proj"):
        if suffix == "trellis":
            return slice(None), slice(first // 16, last // 16), slice(None)
        if suffix == "svh":
            return slice(first, last)
    elif suffix == "trellis":
        return slice(first // 16, last // 16), slice(None), slice(None)
    elif suffix == "suh":
        return slice(first, last)
    return None


def validate_tensors(manifest, layer, expert_ids):
    ids = tuple(expert_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("EXL3 expert IDs must be nonempty and distinct")
    for expert in ids:
        for name, expected in tensor_contract(layer, expert).items():
            actual = manifest.tensors.get(name)
            if actual is None or (actual.dtype, actual.shape) != expected:
                raise ValueError(f"{name}: EXL3 tensor geometry or dtype differs from {expected}")
    return ids


def row_slices(rows):
    if type(rows) is not int or rows <= 0:
        raise ValueError("EXL3 row capacity must be a positive integer")
    return tuple((first, min(first + MAX_ROWS, rows)) for first in range(0, rows, MAX_ROWS))
