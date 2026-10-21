"""Record the deployment's original thin/E3 experts; compare cooperative slices.

`record` runs in the original vLLM image, with its exact EXL3 environment. It
calls the original expert helper, without a server, and saves immutable CPU
inputs, two distinct TP2 expert-slice hashes, and repeated BF16 outputs. An
optional --library also measures the proposed cooperative composition eagerly
in that image. `check` runs the real Flare loader, full-forward compilation,
capture and native worker replay. It does not export/reload a model package or
qualify full-model, TP collectives, routing selection, or serving performance.

Differences are measurements unless explicit --rtol/--atol are supplied. No
reference helper is loaded by the production server.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path

ROWS = (1, 32, 33, 65, 4096)
EXPERT_IDS = (0, 287)


def _device():
    import torch

    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ValueError("assign CUDA_VISIBLE_DEVICES explicitly")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability(0) != (12, 1):
        raise ValueError("the deployment comparison requires SM121")
    return torch.device("cuda:0")


def _hashes(experts):
    import torch

    return {f"{index}.{name}": hashlib.sha256(tensor.detach().contiguous().view(torch.uint8)
        .cpu().numpy().tobytes()).hexdigest()
        for index, tensors in enumerate(experts) for name, tensor in tensors.items()}


def _source_experts(checkpoint, layer, rank, device):
    from qualify.exl3_package import checkpoint_plan, load_expert

    result = []
    for expert in EXPERT_IDS:
        manifest, contract = checkpoint_plan(checkpoint, layer, expert)
        result.append(load_expert(manifest, contract, rank, device))
    return tuple(result)


def _original_layer(module, experts):
    import torch

    layer = torch.nn.Module()
    for suffix in ("trellis", "suh", "svh"):
        setattr(layer, "w13_" + suffix, torch.stack([torch.stack([
            expert[f"{projection}_proj.{suffix}"] for projection in ("gate", "up")]) for expert in experts]))
        setattr(layer, "w2_" + suffix, torch.stack([expert[f"down_proj.{suffix}"] for expert in experts]))
    layer._exl3_hidden_size, layer._exl3_intermediate_local = 4096, 1024
    layer._exl3_bits, layer._exl3_k_words = 4, 64
    layer._exl3_shared_w13_suh = torch.equal(layer.w13_suh[:, 0], layer.w13_suh[:, 1])
    marker = torch.tensor([-877912083], dtype=torch.int32, device=layer.w13_trellis.device)
    layer._exl3_inners = [{name: module.make_linear_exl3(
        expert[f"{name}_proj.trellis"], expert[f"{name}_proj.suh"], expert[f"{name}_proj.svh"], marker)
        for name in ("gate", "up", "down")} for expert in experts]
    module._record_exl3_fat_resolution(layer)
    if layer._exl3_fat_effective_tier != "grouped":
        raise ValueError("the actual two-expert checkpoint did not resolve to original E3 grouped execution")
    module.build_exl3_fused_state(layer, layer._exl3_inners)
    return layer


def _case(rows, device):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(534287 + rows)
    source = (torch.randn((rows, 4096), generator=generator) * .5).bfloat16().to(device)
    # The unselected expert slots are the original helper's nonlocal sentinel.
    # At 33 rows expert 0 is fat and expert 1 is thin; at 65 both are fat.
    ids = torch.full((rows, 8), 2, dtype=torch.int64, device=device)
    weights = torch.zeros((rows, 8), dtype=torch.float32, device=device)
    ids[:, 0] = 0
    ids[::2, 1] = 1
    weights[:, 0] = .6
    weights[::2, 1] = .4
    return source, ids, weights


def _measure(name, actual, expected, repeated, *, rtol=None, atol=None, stage):
    import torch

    actual, expected, repeated = actual.float().cpu(), expected.float().cpu(), repeated.float().cpu()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all() or not torch.isfinite(repeated).all():
        raise ValueError(f"{name}: nonfinite expert output")
    delta, repeat = actual - expected, repeated - expected
    print(json.dumps(dict(case=name, stage=stage, rows=len(actual),
        max_absolute=float(delta.abs().max()), relative_l2=float(delta.norm() / expected.norm().clamp_min(1e-12)),
        unequal_elements=int((actual != expected).sum()), reference_repeat_max=float(repeat.abs().max()),
        reference_repeat_relative_l2=float(repeat.norm() / expected.norm().clamp_min(1e-12)),
        status="measurement_only" if rtol is None else "checking_explicit_bounds")), flush=True)
    if rtol is not None:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _cooperative(experts, library, source, ids, weights):
    import torch
    from ignition.components.exl3 import Exl3Experts, Exl3Workspace, PreparedCooperativeExl3Experts

    packed = Exl3Experts.from_tensors(experts, Exl3Workspace(library, source.device))
    output = torch.empty_like(source)
    return PreparedCooperativeExl3Experts(source, packed, ids.to(torch.int32), weights, output)


def record(args):
    if args.reference.exists():
        raise FileExistsError(args.reference)
    import torch
    from safetensors.torch import save_file
    from vllm.model_executor.layers.quantization import exl3 as original

    device = _device()
    if (not original.fused_moe_enabled() or not original.grouped_fat_enabled()
            or original.temp_rows_fused() != 32 or original.fused_moe_row_tile_enabled()
            or original.exl3_moe_fast_requested() or original.fat_kernel_enabled()
            or original.sorted_fat_fallback_enabled() or original.batched_fat_fallback_enabled()):
        raise ValueError("record requires deployed EXL3 flags: FUSED_MOE=1, FAT_GROUPED=1, TEMP_ROWS_FUSED=32; ROW_TILE/FAT_KERNEL/FAT_SORTED/FAT_BATCHED/GLM53_EXL3_MOE_FAST=0")
    # The cooperative installer preserves this exact callable before wrapping it.
    apply = getattr(original, "_glm53_coop_original_apply", original.apply_exl3_fused_moe)
    if Path(apply.__code__.co_filename).resolve() != Path(original.__file__).resolve():
        raise ValueError("reference callable is not the original module's thin/E3 implementation")
    with torch.inference_mode():
        experts = _source_experts(args.checkpoint, args.layer, args.tp_rank, device)
        hashes = _hashes(experts)
        layer = _original_layer(original, experts)
        tensors, cases = {}, []
        for rows in args.rows:
            name = f"rows{rows}"
            source, ids, weights = _case(rows, device)
            before = original.exl3_fat_diag()["grouped_calls"]
            expected = apply(source, ids, weights, layer, layer._exl3_inners, None, 10.).bfloat16().cpu().contiguous()
            repeated = apply(source, ids, weights, layer, layer._exl3_inners, None, 10.).bfloat16().cpu().contiguous()
            grouped = original.exl3_fat_diag()["grouped_calls"] - before
            if grouped != (2 if rows > 32 else 0):
                raise RuntimeError(f"{name}: expected two original E3 calls for prefill; got {grouped}")
            if not torch.isfinite(expected).all() or not torch.isfinite(repeated).all():
                raise RuntimeError(f"{name}: original E3 output is nonfinite")
            tensors.update({name + ".source": source.cpu(), name + ".ids": ids.cpu(), name + ".weights": weights.cpu(),
                name + ".expected": expected, name + ".repeat": repeated})
            cases.append(name)
            print(json.dumps(dict(case=name, original_grouped_calls=grouped, repeat_max=float((repeated.float()-expected.float()).abs().max()))), flush=True)
            if args.library is not None:
                call = _cooperative(experts, args.library, source, ids, weights)
                call()
                _measure(name, call.output, expected, repeated, stage="eager_cooperative")
                del call
        save_file(tensors, str(args.reference), metadata={"cases": json.dumps(cases), "layer": str(args.layer),
            "tp_rank": str(args.tp_rank), "expert_ids": json.dumps(EXPERT_IDS), "tensor_hashes": json.dumps(hashes),
            "torch": torch.__version__, "source_sha256": hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
            "reference": "original_exl3_moe_thin_plus_E3_temp32", "checkpoint": str(args.checkpoint)})


def check(args):
    import torch
    from safetensors import safe_open
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.recipes.glm53.weights import GlmWeightLoader
    from ignition.components.exl3 import PreparedExl3Experts

    device = _device()
    with torch.inference_mode(), safe_open(args.reference, framework="pt", device="cpu") as reference, contextlib.ExitStack() as resources:
        metadata = reference.metadata()
        if metadata.get("reference") != "original_exl3_moe_thin_plus_E3_temp32" or tuple(json.loads(metadata["expert_ids"])) != EXPERT_IDS:
            raise ValueError("reference is not the distinct-expert original thin/E3 fixture")
        rank, layer = int(metadata["tp_rank"]), int(metadata["layer"])
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        with GlmWeightLoader(inspect_indexed_checkpoint(args.checkpoint), device,
                shard=TensorParallelShard(rank, 2), prefixes=tuple(f"{prefix}.{expert}." for expert in EXPERT_IDS),
                exl3_library=args.library, exl3_prefill_library=args.prefill_library,
                exl3_prefill_capacity=max(args.rows)) as loader:
            experts = loader.load_experts(layer, expert_ids=EXPERT_IDS)
        resources.callback(experts.workspace.close)
        if _hashes(experts.tensors) != json.loads(metadata["tensor_hashes"]):
            raise ValueError("Flare loader changed the reference's original TP2 expert slices")
        for name in json.loads(metadata["cases"]):
            if int(name.removeprefix("rows")) not in args.rows:
                continue
            source = reference.get_tensor(name + ".source").to(device)
            ids = reference.get_tensor(name + ".ids").to(device=device, dtype=torch.int32)
            weights = reference.get_tensor(name + ".weights").to(device)
            output = torch.empty_like(source)
            call = PreparedExl3Experts(source, experts, ids, weights, output)
            compiled = compile_forward(call)
            compiled()
            torch.cuda.synchronize()
            graph, worker = torch.cuda.CUDAGraph(), None
            try:
                with torch.cuda.graph(graph):
                    compiled()
                worker = PreparedGraphBundle(device=0, graphs={"experts": graph},
                    buffers=dict(source=source, ids=ids, weights=weights, output=output),
                    resources=(experts, call, compiled)).start_worker()
                torch.cuda.synchronize()
                counters = ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed")
                before = tuple(torch.cuda.memory_stats()[key] for key in counters)
                ticket = worker.submit(0, repetitions=3)
                completed = worker.poll(timeout_ms=120000)
                assert completed is not None and completed.ticket == ticket and completed.error is None
                after = tuple(torch.cuda.memory_stats()[key] for key in counters)
                if before != after:
                    raise RuntimeError(f"{name}: native replay allocated or freed CUDA storage")
                _measure(name, output, reference.get_tensor(name + ".expected"), reference.get_tensor(name + ".repeat"),
                    rtol=None if args.measure_only else args.rtol, atol=args.atol, stage="compiled_native_replay")
            finally:
                if worker is not None:
                    worker.close()
                graph.reset()


class _PackageCall:
    """The original-prefill specialization of the existing fixed-package probe."""

    def __init__(self, args, manifest, device):
        import torch
        from ignition.components.exl3 import PreparedExl3Experts
        from ignition.components.parallel import TensorParallelShard
        from ignition.preparation.compilation import compile_forward
        from ignition.recipes.glm53.weights import GlmWeightLoader

        expert_ids = (args.expert, 287 if args.expert != 287 else 0)
        prefix = f"model.language_model.layers.{args.layer}.mlp.experts"
        with GlmWeightLoader(manifest, device, shard=TensorParallelShard(args.tp_rank, 2),
                prefixes=tuple(f"{prefix}.{expert}." for expert in expert_ids),
                exl3_library=args.library, exl3_prefill_library=args.prefill_library,
                exl3_prefill_capacity=args.rows) as loader:
            self.weights = loader.load_experts(args.layer, expert_ids=expert_ids)
        self.source, ids, weights = _case(args.rows, device)
        self.output = torch.empty_like(self.source)
        self.forward = PreparedExl3Experts(self.source, self.weights, ids.to(torch.int32), weights, self.output)
        self.compiled = compile_forward(self.forward)
        self.counters = self.weights.workspace.prefill.locks
        self.scratch = self.weights.workspace.tensors()

    def __call__(self):
        self.compiled()

    def close(self):
        self.weights.workspace.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("record", "check"):
        command = sub.add_parser(mode)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--reference", type=Path, required=True)
        command.add_argument("--library", type=Path, required=mode == "check")
        command.add_argument("--rows", type=int, choices=ROWS, nargs="+", default=ROWS)
        if mode == "record":
            command.add_argument("--layer", type=int, choices=range(3, 45), default=3)
            command.add_argument("--tp-rank", type=int, choices=(0, 1), required=True)
        else:
            command.add_argument("--prefill-library", type=Path, required=True)
            command.add_argument("--measure-only", action="store_true")
            command.add_argument("--rtol", type=float)
            command.add_argument("--atol", type=float)
    args = parser.parse_args(argv)
    if args.mode == "check" and not args.measure_only:
        if args.rtol is None or args.atol is None or not 0 <= args.rtol < float("inf") or not 0 <= args.atol < float("inf"):
            parser.error("set finite nonnegative --rtol and --atol from the required arithmetic and repeatability, or use --measure-only")
    {"record": record, "check": check}[args.mode](args)


if __name__ == "__main__":
    main()
