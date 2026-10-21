"""Canonical GLM 5.3 Flash structure from a local checkpoint configuration."""

import json
from pathlib import Path


def validate_linear_attention_config(text):
    """Accept the original nested schema and older flattened ModelOpt metadata."""
    from collections.abc import Mapping

    nested = text.get("linear_attn_config", {})
    if not isinstance(nested, Mapping):
        raise ValueError("GLM linear_attn_config must be a mapping")
    for flat, key, expected in (
        ("linear_num_heads", "num_heads", 64),
        ("linear_head_dim", "head_dim", 128),
        ("linear_conv_kernel_dim", "short_conv_kernel_size", 4),
    ):
        values = [mapping[name] for mapping, name in ((text, flat), (nested, key)) if name in mapping]
        if not values or any(type(value) is not int or value != expected for value in values):
            raise ValueError(f"GLM {flat}/linear_attn_config.{key} requires {expected}, got {values}")


def create_model_tree(snapshot: Path):
    """Construct the canonical tree on meta storage, without checkpoint transfers.

    Packed weight loading, tensor-parallel partitioning, and prepared component
    replacement must precede execution. This function does not initialize CUDA
    or provide a serving forward.
    """
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
    )

    from ignition.components.model import canonical_tree

    raw = json.loads((Path(snapshot) / "config.json").read_text())
    if (raw.get("model_type") != "glm5_next"
            or raw.get("architectures") != ["Glm5NextForConditionalGeneration"]):
        raise ValueError("GLM 5.3 Flash preparation requires Glm5NextForConditionalGeneration")
    return canonical_tree(Glm5NextForConditionalGeneration, Glm5NextConfig.from_dict(raw))
