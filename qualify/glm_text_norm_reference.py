"""Compare BF16 text RMSNorm with the deployment's actual CUDA implementation.

Record uses the original deployment image. Check compiles the recipe's precision
selection, captures it, and replays through the native worker. This isolates norm
arithmetic; it does not qualify the mHC mixing or the complete text decoder.
"""

import argparse
import json
from pathlib import Path

NAMES = (
    "model.language_model.layers.3.self_attn.q_a_layernorm.weight",
    "model.language_model.layers.3.self_attn.kv_a_layernorm.weight",
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.post_attention_layernorm.weight",
    "model.language_model.norm.weight",
)
ROWS = (1, 17, 65)


def record(args):
    import torch
    import vllm._C_stable_libtorch  # noqa: F401; registers the actual deployed CUDA op
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint = args.checkpoint
    config = json.loads((checkpoint / "config.json").read_text())
    epsilon = config["text_config"]["rms_norm_eps"]
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tensors, metadata = {}, {"epsilon": str(epsilon), "names": json.dumps(NAMES)}
    for index_number, name in enumerate(NAMES):
        with safe_open(checkpoint / index[name], framework="pt", device="cpu") as shard:
            weight = shard.get_tensor(name)
        assert weight.dtype == torch.bfloat16 and weight.ndim == 1
        tensors[f"weight.{index_number}"] = weight
        weight_gpu = weight.cuda()
        for rows in ROWS:
            # BF16 inputs are stored, so both Torch versions receive identical bytes.
            values = torch.arange(rows * len(weight), dtype=torch.float32).view(rows, -1)
            source = (values.mul(.017).sin() * 3 + values.mul(.003).cos()).bfloat16().cuda()
            output, repeated = torch.empty_like(source), torch.empty_like(source)
            torch.ops._C.rms_norm(output, source, weight_gpu, epsilon)
            torch.ops._C.rms_norm(repeated, source, weight_gpu, epsilon)
            torch.cuda.synchronize()
            torch.testing.assert_close(output, repeated, rtol=0, atol=0)
            key = f"{index_number}.{rows}"
            tensors[f"source.{key}"] = source.cpu()
            tensors[f"expected.{key}"] = output.cpu()
        print(json.dumps(dict(recorded=name, dtype=str(weight.dtype), width=len(weight), epsilon=epsilon)), flush=True)
    save_file(tensors, args.reference, metadata=metadata)


class _Norms:
    def __init__(self, items, kernels):
        self.items, self.kernels = items, kernels

    def __call__(self):
        from ignition.recipes.glm53.projections import rms_norm

        for source, weight, output, epsilon in self.items:
            rms_norm(source, weight, output, epsilon, projection_weight=weight, kernels=self.kernels)


def check(args):
    import torch
    from flashinfer.triton.norm import rms_norm as legacy_rms_norm
    from safetensors import safe_open

    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.forward_metadata import dumps, loads
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.components.glm_vision_kernels import OriginalVisionKernels

    torch.cuda.set_device(0)
    items, expected, names = [], [], []
    with safe_open(args.reference, framework="pt", device="cuda:0") as file:
        metadata = file.metadata()
        epsilon = float(metadata["epsilon"])
        for index, name in enumerate(json.loads(metadata["names"])):
            weight = file.get_tensor(f"weight.{index}")
            for rows in ROWS:
                source = file.get_tensor(f"source.{index}.{rows}")
                if len(weight) in (512, 1536):
                    # The real MLA low-rank inputs are strided views of its fused projection.
                    storage = torch.full((rows, 2208), float("nan"), dtype=source.dtype, device=source.device)
                    start = 1536 if len(weight) == 512 else 0
                    storage[:, start:start + len(weight)].copy_(source)
                    source = storage[:, start:start + len(weight)]
                reference = file.get_tensor(f"expected.{index}.{rows}")
                output = torch.empty_like(source)
                items.append((source, weight, output, epsilon))
                expected.append(reference)
                names.append((name, rows))
                values = source.float()
                single = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
                    * weight.float()).bfloat16()
                old = torch.empty_like(source)
                legacy_rms_norm(source, weight, old, epsilon, round_before_weight=True)
                print(json.dumps(dict(name=name, rows=rows,
                    original_vs_fp32_single_cast_max=float((reference-single).abs().max()),
                    legacy_round_before_weight_max=float((reference-old).abs().max()))), flush=True)
    kernels = OriginalVisionKernels(args.kernels, torch.device("cuda:0"))
    forward = _Norms(tuple(items), kernels)
    traced = loads(dumps(forward))
    assert traced.kernels.handle == kernels.handle and len(traced.items) == len(items)
    compiled = compile_forward(forward)
    compiled()
    graph, worker = torch.cuda.CUDAGraph(), None
    try:
        with torch.cuda.graph(graph):
            compiled()
        worker = PreparedGraphBundle(device=0, graphs={"norms": graph}, buffers={f"out.{index}": item[2] for index, item in enumerate(items)},
            resources=(forward, compiled)).start_worker()
        for _, _, output, _ in items:
            output.fill_(float("nan"))
        torch.cuda.synchronize()
        keys = ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed")
        before = tuple(torch.cuda.memory_stats()[key] for key in keys)
        ticket = worker.submit(0, repetitions=3)
        completed = worker.poll(timeout_ms=120000)
        assert completed is not None and completed.ticket == ticket and completed.error is None
        assert before == tuple(torch.cuda.memory_stats()[key] for key in keys)
        for item, reference, (name, rows) in zip(items, expected, names, strict=True):
            output = item[2]
            print(json.dumps(dict(checked=name, rows=rows,
                max_abs=float((output-reference).abs().max()), unequal=int((output != reference).sum()))), flush=True)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
    finally:
        if worker is not None:
            worker.close()
        graph.reset()
        kernels.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    recording = subparsers.add_parser("record")
    recording.add_argument("--checkpoint", type=Path, required=True)
    recording.add_argument("--reference", type=Path, required=True)
    checking = subparsers.add_parser("check")
    checking.add_argument("--reference", type=Path, required=True)
    checking.add_argument("--kernels", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "record" and args.reference.exists():
        raise FileExistsError(args.reference)
    (record if args.mode == "record" else check)(args)


if __name__ == "__main__":
    main()
