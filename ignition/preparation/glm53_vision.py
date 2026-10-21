"""Torch-free geometry and checkpoint precision of the deployed GLM vision tower."""


def vision_config(raw):
    c = dict(raw["vision_config"])
    for key, value in dict(hidden_size=1024, num_heads=16, depth=24, intermediate_size=4096,
            out_hidden_size=4096, projection_intermediate_size=10240, patch_size=14,
            temporal_patch_size=2, spatial_merge_size=2, in_channels=3, hidden_act="silu").items():
        if c.get(key) != value:
            raise ValueError(f"unsupported GLM vision {key}: expected {value}, got {c.get(key)}")
    c["swiglu_limit"] = c.get("swiglu_limit", raw["text_config"].get("swiglu_limit"))
    if c.get("rms_norm_eps") != 1e-5 or c["swiglu_limit"] != 10.:
        raise ValueError("GLM vision requires checkpoint RMSNorm epsilon 1e-5 and SwiGLU limit 10")
    # The deployed Glm5NextConfig constructs its nested vision config with the
    # canonical 1e-6 block/post RMSNorm epsilon. This differs from the raw file's
    # mirrored text epsilon; q/k RMSNorm remains independently fixed at 1e-5.
    c["block_norm_eps"] = 1e-6
    return c


def tensor_contract():
    result = {"patch_embed.proj.weight": (1024, 3, 2, 14, 14), "patch_embed.proj.bias": (1024,),
        "post_layernorm.weight": (1024,), "downsample.weight": (4096, 1024, 2, 2),
        "downsample.bias": (4096,), "merger.proj.weight": (4096, 4096),
        "merger.post_projection_norm.weight": (4096,), "merger.post_projection_norm.bias": (4096,),
        "merger.gate_proj.weight": (10240, 4096), "merger.up_proj.weight": (10240, 4096),
        "merger.down_proj.weight": (4096, 10240)}
    for layer in range(24):
        p = f"blocks.{layer}."
        result.update({p + name: shape for name, shape in {
            "norm1.weight": (1024,), "norm2.weight": (1024,), "attn.q_norm.weight": (64,),
            "attn.k_norm.weight": (64,), "attn.qkv.weight": (3072, 1024), "attn.qkv.bias": (3072,),
            "attn.proj.weight": (1024, 1024), "attn.proj.bias": (1024,),
            "mlp.gate_proj.weight": (4096, 1024), "mlp.gate_proj.bias": (4096,),
            "mlp.up_proj.weight": (4096, 1024), "mlp.up_proj.bias": (4096,),
            "mlp.down_proj.weight": (1024, 4096), "mlp.down_proj.bias": (1024,)}.items()})
    return {"model.visual." + name: ("BF16", shape) for name, shape in result.items()}


def validate_tensors(manifest):
    expected = tensor_contract()
    for name, contract in expected.items():
        tensor = manifest.tensors.get(name)
        if tensor is None or (tensor.dtype, tensor.shape) != contract:
            raise ValueError(f"{name}: GLM vision requires stored {contract}")
    extra = {name for name in manifest.tensors if name.startswith("model.visual.")} - expected.keys()
    if extra:
        raise ValueError(f"unsupported GLM vision tensor: {min(extra)}")
