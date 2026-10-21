"""Record the deployed vLLM vision tower, then measure Flare native replay.

Run `record` in the original vLLM environment and `check` in Flare's preparation
one. `--measure-only` reports differences without claiming numerical qualification.
No Python reference path is used by the production server.
"""

import argparse
import json
from pathlib import Path
import tempfile


def record(args):
    import torch
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import save_file
    import vllm.models
    if args.models_path is not None:
        vllm.models.__path__.insert(0, str(args.models_path))
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.models.glm5next.nvidia.multimodal import Glm5NextVisionTransformer
    from vllm.transformers_utils.processors.glm5next import Glm5NextImageProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("record requires an explicitly assigned CUDA GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    config = VllmConfig(model_config=ModelConfig(model=str(args.checkpoint), dtype="bfloat16",
        trust_remote_code=True, max_model_len=4096))
    config.kernel_config.ir_op_priority.set_default()
    rendezvous = tempfile.mktemp(prefix="flare-glm-vision-dist-")
    with set_current_vllm_config(config):
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
            distributed_init_method="file://" + rendezvous)
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.inference_mode(), set_current_vllm_config(config), torch.device(device):
            from vllm.model_executor.layers.layernorm import RMSNorm
            normalizer = RMSNorm(1024, eps=1e-5).to(device)
            sample = torch.linspace(-7, 8, 1024, device=device).sin().view(1, -1)
            normalizer.weight.copy_(torch.linspace(.5, 1.5, 1024, device=device))
            actual = normalizer(sample)
            normalized = sample.float() * torch.rsqrt(sample.float().square().mean(-1, keepdim=True) + 1e-5)
            single = (normalized * normalizer.weight.float()).bfloat16()
            rounded = normalized.bfloat16() * normalizer.weight
            print(json.dumps(dict(norm_microcheck="actual_cuda_provider", single_cast_max=float((actual-single).abs().max()),
                round_before_weight_max=float((actual-rounded).abs().max()))), flush=True)
            del normalizer, sample, actual, normalized, single, rounded
            hf = config.model_config.hf_config
            module = Glm5NextVisionTransformer(hf.text_config, hf.vision_config,
                norm_eps=hf.vision_config.rms_norm_eps, quant_config=None).eval()
            print(json.dumps(dict(rotary_cache_dtype=str(module.rotary_pos_emb.cos_sin_cache.dtype))), flush=True)
        index_path = args.checkpoint / "model.safetensors.index.json"
        shards = (sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
            if index_path.is_file() else ["model.safetensors"])
        def weights():
            for shard in shards:
                with safe_open(args.checkpoint / shard, framework="pt", device="cpu") as source:
                    for name in source.keys():
                        if name.startswith("model.visual."):
                            yield name.removeprefix("model.visual."), source.get_tensor(name)
        with torch.inference_mode(), set_current_vllm_config(config):
            module.load_weights(weights())
        options = json.loads((args.checkpoint / "processor_config.json").read_text())["image_processor"]
        processor = Glm5NextImageProcessor(**{**options, "max_image_tokens": 2048})
        tensors, cases = {}, []
        sizes = [(112, 112), (29, 30), (224, 224)] + ([(4000, 3000)] if args.large else [])
        for index, (width, height) in enumerate(sizes):
            name = f"case{index}"
            raw = bytes((x * 17 + y * 7 + channel * 43) % 256
                for y in range(height) for x in range(width) for channel in range(3))
            result = processor(images=[Image.frombytes("RGB", (width, height), raw)], return_tensors="pt")
            grid, pixels = result["image_grid_thw"], result["pixel_values"]
            _, rows, columns = grid[0].tolist()
            positions = torch.stack(torch.meshgrid(torch.arange(rows), torch.arange(columns), indexing="ij"), -1)
            positions = positions.reshape(rows // 2, 2, columns // 2, 2, 2).permute(0, 2, 1, 3, 4).reshape(-1, 2)
            traces, handles = {}, []
            if args.trace is not None and index == args.trace_case:
                from qualify.glm_vision_trace import install_trace
                handles = install_trace(module, traces, args.trace_layer)
            with torch.inference_mode(), set_current_vllm_config(config):
                encoded_pixels = pixels.to(device)
                metadata = None
                if args.padding_capacity is not None:
                    capacity = args.padding_capacity
                    if capacity < len(pixels) or capacity % 4:
                        raise ValueError("padding capacity must cover every patch and be divisible by four")
                    encoded_pixels = torch.nn.functional.pad(encoded_pixels, (0, 0, 0, capacity - len(pixels)))
                    metadata = module.prepare_encoder_metadata(grid.tolist(), max_seqlen_override=capacity)
                    for key in ("rotary_pos_emb_cos", "rotary_pos_emb_sin"):
                        metadata[key] = torch.nn.functional.pad(metadata[key], (0, 0, 0, capacity-len(pixels)),
                            value=1. if key.endswith("cos") else 0.)
                expected = module(encoded_pixels, grid, encoder_metadata=metadata)[:len(pixels)//4].cpu().contiguous()
                repeated = module(encoded_pixels, grid, encoder_metadata=metadata)[:len(pixels)//4].cpu().contiguous()
            for handle in handles:
                handle.remove()
            if traces:
                traces.update(pixels=pixels.cpu().contiguous(), positions=positions, grid=grid.contiguous(), expected=expected)
                save_file(traces, str(args.trace), metadata={"case": name})
            tensors.update({f"{name}.pixels": pixels.cpu().contiguous(), f"{name}.positions": positions,
                f"{name}.grid": grid.contiguous(), f"{name}.expected": expected, f"{name}.repeat": repeated})
            cases.append(name)
            print(f"{name}: {width}x{height}, {len(pixels)} patches, output {list(expected.shape)}", flush=True)
        save_file(tensors, str(args.reference), metadata={"cases": json.dumps(cases), "torch": torch.__version__,
            "checkpoint": str(args.checkpoint), "encoder": type(module).__name__,
            "padding_capacity": str(args.padding_capacity)})
    finally:
        torch.set_default_dtype(previous)
        from vllm.distributed import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel()
        destroy_distributed_environment()


def check(args):
    import torch
    from safetensors import safe_open
    from b12x.preparation import PreparationSession
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.glm53.vision import PATCH_CAPACITIES, PreparedVision, VisionInputs, VisionScratch
    from ignition.recipes.glm53.vision_weights import VisionWeights
    from ignition.components.glm_vision_kernels import OriginalVisionKernels

    if not args.measure_only and (args.rtol is None or args.atol is None):
        raise ValueError("set explicit --rtol and --atol from the required arithmetic and measured repeatability, or use --measure-only")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    weights = VisionWeights.load(inspect_indexed_checkpoint(args.checkpoint), device)
    kernels = OriginalVisionKernels(args.kernels, device)
    inputs = VisionInputs.allocate(device)
    with safe_open(args.reference, framework="pt", device="cpu") as source:
        cases = json.loads(source.metadata()["cases"])
        capacities = {name: args.capacity or next(n for n in PATCH_CAPACITIES
            if n >= source.get_tensor(name + ".pixels").shape[0]) for name in cases}
        if any(capacity not in PATCH_CAPACITIES or source.get_tensor(name + ".pixels").shape[0] > capacity
                for name, capacity in capacities.items()):
            raise ValueError("explicit capacity must cover every case and belong to the prepared catalog")
        for capacity in sorted(set(capacities.values())):
            arena = ScratchArena(device)
            scratch = VisionScratch.allocate(capacity, device)
            with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
                encoder = PreparedVision(weights, scratch, inputs, kernels=kernels, session=session, workspace=arena, name="glm-vision-reference")
                arena.finalize()
                call = compile_forward(encoder)
                call()
                session.freeze()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    call()
                worker = PreparedGraphBundle(device=0, graphs={"vision": graph}, buffers={},
                    resources=(weights, scratch, inputs, encoder, call, session, arena)).start_worker()
                try:
                    for name in (name for name in cases if capacities[name] == capacity):
                        pixels = source.get_tensor(name + ".pixels")
                        expected = source.get_tensor(name + ".expected").float()
                        repeated = source.get_tensor(name + ".repeat").float()
                        count = pixels.shape[0]
                        scratch.patches.zero_()
                        scratch.patches[:count].copy_(pixels)
                        encoder.positions.zero_()
                        encoder.positions[:count].copy_(source.get_tensor(name + ".positions"))
                        encoder.cu.copy_(torch.tensor([0, count], dtype=torch.int32, device=device))
                        encoder.num_patches.fill_(count)
                        encoder.num_features.fill_(count // 4)
                        eager = encoder()[:count // 4].float().cpu() if args.compare_eager else None
                        torch.cuda.synchronize()
                        allocation = torch.cuda.memory_stats()["allocation.all.allocated"]
                        ticket = worker.submit(0)
                        completion = worker.poll(timeout_ms=60000)
                        assert completion is not None and completion.ticket == ticket and completion.error is None
                        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocation
                        actual = scratch.output[:count // 4].float().cpu()
                        assert torch.isfinite(actual).all()
                        delta, repeat = actual - expected, repeated - expected
                        print(json.dumps(dict(case=name, patches=count, capacity=capacity, relative_l2=float(delta.norm() / expected.norm().clamp_min(1e-12)),
                            max_absolute=float(delta.abs().max()), reference_repeat_max=float(repeat.abs().max()),
                            status="measurement_only" if args.measure_only else "checking_explicit_bounds")), flush=True)
                        if eager is not None:
                            print(json.dumps(dict(case=name, eager_relative_l2=float((eager-expected).norm()/expected.norm()),
                                native_vs_eager_relative_l2=float((actual-eager).norm()/eager.norm()),
                                native_vs_eager_max=float((actual-eager).abs().max()))), flush=True)
                        if not args.measure_only:
                            torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
                finally:
                    worker.close()
                    graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("record", "check"):
        command = sub.add_parser(mode)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--reference", type=Path, required=True)
        if mode == "record":
            command.add_argument("--large", action="store_true")
            command.add_argument("--trace", type=Path)
            command.add_argument("--trace-case", type=int, default=0)
            command.add_argument("--trace-layer", type=int, default=0)
            command.add_argument("--padding-capacity", type=int)
            command.add_argument("--models-path", type=Path)
        else:
            command.add_argument("--measure-only", action="store_true")
            command.add_argument("--compare-eager", action="store_true")
            command.add_argument("--kernels", type=Path, required=True)
            command.add_argument("--capacity", type=int)
            command.add_argument("--rtol", type=float)
            command.add_argument("--atol", type=float)
    args = parser.parse_args()
    {"record": record, "check": check}[args.mode](args)


if __name__ == "__main__":
    main()
