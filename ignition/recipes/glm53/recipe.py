"""Qualified checkpoint, serving range, and memory costs of the GLM 5.3 recipe."""

from types import MappingProxyType
from pathlib import Path
import hashlib
from dataclasses import replace
import json
import math

from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange, Speculation

# Each sparse-attention layer stores four latent blocks and one index block per
# 256-token page on every rank. The target has eleven such layers; learned drafts
# add the draft layer's.
_PAGE_LAYER_BYTES = 4 * 64 * 528 + 64 * 132
_TARGET_PAGE_LAYERS = 11
# Rough estimate of what a rank holds beyond stored weights, state rows, and
# pages: operator scratch, graphs, collectives, and the CUDA context. It serves
# every rank count and speculation setting.
_GRAPH_BYTES = 4864 << 20


def _costs(state_row_bytes, graph_bytes, *, ranks, speculating):
    layers = _TARGET_PAGE_LAYERS + int(speculating)
    # Decode context parallelism shards each page's latent records over the ranks.
    return RankCosts(state_row_bytes, layers * _PAGE_LAYER_BYTES, graph_bytes,
                     draft_page_bytes=_PAGE_LAYER_BYTES if speculating else 0, page_tokens=256 * ranks)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    """Prepare the plan's catalog and speculation settings on process-local GPUs 0 through N-1.

    With `ranks`, rank zero's multi-host rendezvous, each rank uses its own host's GPU 0.
    KV pages fill the free memory measured after graph capture, not the plan's estimate.
    """
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_token_capacity": plan.speculative_token_capacity, "speculative_fixed_width": speculation.fixed_width}
    from .dflash_weights import DRAFT_DIRECTORY
    draft_checkpoint = Path(snapshot) / DRAFT_DIRECTORY
    if speculation is not None and draft_checkpoint.is_dir():
        drafting["dflash_checkpoint"] = draft_checkpoint
    config = PreparationConfig(image_concurrency=plan.image_concurrency, compile_workers=plan.compile_workers, tuning_replica=plan.tuning_replica,
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts, compile_decoder=compile_forwards, retain_graphs=retain_graphs,
        max_active=plan.concurrency, roce_ring=plan.roce_ring, context_parallel=plan.context_parallel, stream_overlap=plan.stream_overlap,
        context=plan.recipe.context, prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, kernel_cache=kernel_cache,
        # SM103 runs routed experts through trtllm-gen with NVFP4 activations at every size.
        expert_activation="a4" if any(gpu.capability == (10, 3) for gpu in plan.gpus) else "a16",
        # Expanded scales take 2 x 288 x 4096 x 2048 / 16 bytes per routed layer over the ranks: about
        # 4.9 GB per rank at TP4, which GB10 holds beside its weights, but 9.7 GB at TP2, which it does not.
        expand_expert_scales=plan.world_size >= 4,
        exl3_library=plan.exl3_library, exl3_prefill_library=plan.exl3_prefill_library,
        glm_vision_kernels=plan.glm_vision_kernels, mtp=speculation is not None, **drafting)
    if plan.world_size == 1:
        return prepare_rank(snapshot, config, rank=0, group=None, progress=progress, package_storage=package_storage)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="glm53", served_model_name="GLM-5.3-Flash", prepare=prepare, usage=usage,
    # Bounds span HLE, BabyVision, and coding presets: https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1,
        # Zero is an explicit migration requirement; positive bounds retain the model-card presets.
        ranges={"temperature": SamplingRange(0., 1.), "top_p": SamplingRange(.95, 1.)}),
    checkpoints=(Checkpoint(
        alias="glm-5.3-flash", repository="local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
        revision="dec48abd33efa73c3bb7c95b74eee10cad34f9be", architecture="Glm5NextForConditionalGeneration",
        stored_bytes=178_175_535_810,
        # Language model, draft layer and output head; this pinned recipe remains text-only.
        device_bytes=177_784_906_130,
        # Decoder layer 45, the MTP module.
        draft_bytes=4_301_658_112),),
    context=1048576, page_tokens=256, concurrency=4, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=5), max_draft_tokens=7, verification_rows=0,
    # Shared experts, KDA gate projections, and decode query chains run on an auxiliary stream.
    stream_overlap=True,
    ranks=MappingProxyType({
        # One rank holds every head: twice TP2's recurrent rows less the replicated part.
        1: _costs(161_123_584, _GRAPH_BYTES, ranks=1, speculating=True),
        2: _costs(81_464_576, _GRAPH_BYTES, ranks=2, speculating=True),
        4: _costs(41_635_072, _GRAPH_BYTES, ranks=4, speculating=True),
    }),
    plain_ranks=MappingProxyType({
        1: _costs(149_230_592, _GRAPH_BYTES, ranks=1, speculating=False),
        2: _costs(75_420_672, _GRAPH_BYTES, ranks=2, speculating=False),
        4: _costs(38_515_712, _GRAPH_BYTES, ranks=4, speculating=False),
    }),
)


def _resident_exl3_bytes(tensor):
    """Bytes retained on one TP2 rank, following GlmWeightLoader's actual slices."""
    name, size = tensor.name, tensor.nbytes
    if name.startswith("model.visual.") or name in (
            "model.language_model.embed_tokens.weight", "model.language_model.norm.weight"):
        return size
    if name == "lm_head.weight":
        return size // 2
    prefix = "model.language_model.layers."
    if not name.startswith(prefix):
        raise ValueError(f"unsupported GLM EXL3 tensor: {name}")
    layer, _, path = name.removeprefix(prefix).partition(".")
    if not layer.isdigit() or not 0 <= int(layer) <= 45:
        raise ValueError(f"invalid GLM layer tensor: {name}")
    if int(layer) == 45:
        return 0  # This EXL3 target reader does not prepare an MTP draft.
    if ".mlp.experts." in name:
        projection, suffix = name.rsplit(".", 2)[-2:]
        if suffix == "mcg":
            return 0  # The scalar marker is checked during loading, not retained.
        replicated = (projection in ("gate_proj", "up_proj") and suffix == "suh"
                      or projection == "down_proj" and suffix == "svh")
        return size if replicated else size // 2
    if path.startswith("hc_"):
        return math.prod(tensor.shape) * 4  # mHC executes its checkpoint gains in FP32.
    if path.startswith("mlp."):
        return size if path.startswith("mlp.gate.") else size // 2
    if path.startswith("self_attn."):
        parameter = path.removeprefix("self_attn.").split(".")[0]
        sharded = ({"q_b_proj", "kv_b_proj", "o_proj"} if int(layer) % 4 == 3 else
            {"q_proj", "k_proj", "v_proj", "b_proj", "f_b_proj", "g_b_proj", "o_proj",
             "A_log", "dt_bias", "q_conv1d", "k_conv1d", "v_conv1d"})
        return size // 2 if parameter in sharded else size
    if path in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return size
    raise ValueError(f"unsupported GLM EXL3 target tensor: {name}")


def checkpoint_variant(snapshot, recipe, checkpoint):
    """Declare the exact routed-only EXL3/BF16 catalog from real checkpoint headers.

    The public NVFP4 recipe and already-exported packages retain their semantics.
    Local EXL3 resolution requires complete local headers; tensor payload validation
    and numerical qualification still belong to loading and the GPU checks.
    """
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.exl3 import validate_config, validate_tensors
    from ignition.preparation.glm53_vision import vision_config, validate_tensors as validate_vision

    raw = json.loads((snapshot / "config.json").read_text())
    quantization = raw.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "exl3":
        return recipe, checkpoint
    validate_config(raw, 2)
    vision_config(raw)
    manifest = inspect_indexed_checkpoint(snapshot)
    validate_vision(manifest)
    for layer in range(3, 45):
        validate_tensors(manifest, layer, range(288))
    target = [tensor for tensor in manifest.tensors.values()
              if not tensor.name.startswith("model.language_model.layers.45.")]
    for tensor in target:
        if ".mlp.experts." not in tensor.name and tensor.dtype not in ("BF16", "F32"):
            raise ValueError(f"{tensor.name}: routed-only EXL3 requires native BF16/FP32 non-routed storage")
    # Nine rank-local pointer tables per routed layer, in addition to unchanged payloads.
    per_rank = sum(_resident_exl3_bytes(tensor) for tensor in target) + 42 * 288 * 9 * 8
    replacement = replace(checkpoint, stored_bytes=sum(t.nbytes for t in manifest.tensors.values()),
        device_bytes=2 * per_rank, draft_bytes=0, mapped_host_bytes=0, expansion_bytes=0,
        scale_compressed_device_bytes={}, ranks=None, plain_ranks=None)
    costs = MappingProxyType({2: RECIPE.plain_ranks[2]})
    variant = replace(recipe, checkpoints=(replacement,), ranks=costs, plain_ranks=MappingProxyType({}),
        speculation=None, max_draft_tokens=0, max_images=20, image_bytes=2048 * 4096 * 2)
    from .dflash_weights import DRAFT_DIRECTORY, inspect_draft
    draft_path = snapshot / DRAFT_DIRECTORY
    if draft_path.is_dir():
        draft_config, draft_manifest = inspect_draft(draft_path)
        # QKV/output and MLP intermediates partition over TP2; other draft tensors replicate.
        sharded = (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
                   ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
        draft_bytes = sum(t.nbytes // (2 if name.endswith(sharded) else 1)
                          for name, t in draft_manifest.tensors.items())
        # The fused context KV matrix retains a separate copy of each layer's K/V rows.
        draft_bytes += 5 * 2 * 512 * 4096 * 2
        identity = hashlib.sha256((draft_path / "config.json").read_bytes())
        for shard in draft_manifest.shards:
            identity.update(shard.header_sha256.encode())
        replacement = replace(replacement,
            stored_bytes=replacement.stored_bytes + sum(t.nbytes for t in draft_manifest.tensors.values()),
            device_bytes=replacement.device_bytes + 2 * draft_bytes, draft_bytes=2 * draft_bytes,
            revision=replacement.revision + "-dflash2-" + identity.hexdigest())
        # Existing speculative target rows, less the integrated MTP's three state regions,
        # plus five sharded BF16 draft KV windows. Every added region is 256-byte aligned.
        state_bytes = RECIPE.ranks[2].state_row_bytes - (2 * 12 * 128 * 2 + _PAGE_LAYER_BYTES)
        state_bytes += sum(math.prod(shape) * 2 for _, shape, _ in draft_config.regions(2))
        speculative_costs = MappingProxyType({2: RankCosts(state_bytes, _TARGET_PAGE_LAYERS * _PAGE_LAYER_BYTES,
            _GRAPH_BYTES + (512 << 20), page_tokens=512)})
        variant = replace(variant, checkpoints=(replacement,), speculation=Speculation(draft_tokens=7),
            max_draft_tokens=7, ranks=speculative_costs, plain_ranks=costs)
    return variant, replacement
