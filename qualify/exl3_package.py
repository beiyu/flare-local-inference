"""Qualify the deployed GLM EXL3 cooperative kernel through a native package.

This is a GPU mechanism probe, not GLM serving support. One real checkpoint
expert, sharded as TP2, is referenced by eight distinct expert slots with uniform
route weights. No weights are requantized. The probe captures its fixed-buffer
launch, exports with the production allocator and exporter, and optionally runs
the existing Rust fixed-package checker in a fresh process. It does not qualify
complete-forward compilation, prefill, tensor-parallel collectives, or HTTP.

Build the checker before running on an explicitly assigned, idle SM121 GPU:

    cargo build --manifest-path helios/Cargo.toml --no-default-features \\
      --features runtime-package,test-tools --bin test_fixed_runtime_package
    CUDA_VISIBLE_DEVICES=0 python -m qualify.exl3_package run \\
      --checkpoint /workspace/models/GLM-5.3-Flash-EXL3-TR3-4bpw \\
      --library /path/to/cooperative_moe.so --output /unused/probe-directory \\
      --native-executable helios/target/debug/test_fixed_runtime_package

The checker uses public fixture signing credentials and raw native replay. A pass
does not establish numerical agreement with BF16 or another quantization format.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


# The existing validated geometry-1 GLM specialization, not the DS4.1 binary.
LIBRARY_SHA256 = "9b7d4e1fc81402313d82f286f027a067bc2b5b7b0684d066096112585c06b717"
HIDDEN, INTERMEDIATE, TOPK, MAX_ROWS, EXPERT_SLOTS = 4096, 1024, 8, 32, 8
BITS, GEOMETRY, MCG_MARKER = 4, 1, -877912083
SLOTS_MAX = TOPK * MAX_ROWS
COUNTER_ELEMENTS = SLOTS_MAX * (INTERMEDIATE // 128) + MAX_ROWS * (HIDDEN // 128) + 2 + 2 * SLOTS_MAX + 1


def tensor_contract(layer: int, expert: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Full checkpoint geometry, before the exact TP2 expert slices."""
    if type(layer) is not int or not 3 <= layer < 45:
        raise ValueError("layer must identify a routed target layer, 3 through 44")
    if type(expert) is not int or not 0 <= expert < 288:
        raise ValueError("expert must be 0 through 287")
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}"
    result = {}
    for projection, inputs, outputs in (("gate_proj", HIDDEN, 2 * INTERMEDIATE),
                                       ("up_proj", HIDDEN, 2 * INTERMEDIATE),
                                       ("down_proj", 2 * INTERMEDIATE, HIDDEN)):
        for suffix, dtype, shape in (
            ("trellis", "I16", (inputs // 16, outputs // 16, BITS * 16)),
            ("suh", "F16", (inputs,)), ("svh", "F16", (outputs,)),
            ("mcg", "I32", (1,)),
        ):
            result[f"{prefix}.{projection}.{suffix}"] = dtype, shape
    return result


def checkpoint_plan(checkpoint: Path, layer: int, expert: int):
    """Validate real safetensors metadata without importing Torch or reading weights."""
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    config = json.loads((checkpoint / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if (config.get("architectures") != ["Glm5NextForConditionalGeneration"]
            or quant.get("quant_method") != "exl3" or quant.get("bits") != BITS
            or quant.get("codebook") != "mcg" or quant.get("scope") != "glm53_routed_experts_only"):
        raise ValueError("probe requires GLM-5.3 routed-experts-only EXL3 K4 MCG metadata")
    contract = tensor_contract(layer, expert)
    manifest = inspect_indexed_checkpoint(checkpoint)
    for name, (dtype, shape) in contract.items():
        tensor = manifest.tensors.get(name)
        if tensor is None or (tensor.dtype, tensor.shape) != (dtype, shape):
            raise ValueError(f"{name}: expected checkpoint {dtype} {shape}, got {tensor}")
    return manifest, contract


def validate_library(path: Path) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != LIBRARY_SHA256:
        raise ValueError("cooperative library differs from the validated GLM geometry-1 binary")


def load_expert(manifest, contract, tp_rank: int, device):
    """Load only one expert's source tensors, keeping their EXL3 encoding unchanged."""
    import torch
    from safetensors import safe_open
    from ignition.preparation.tensor_metadata import record_numeric_tensor

    if tp_rank not in (0, 1):
        raise ValueError("tp_rank must be 0 or 1")
    tensors = {}
    for name in contract:
        metadata = manifest.tensors[name]
        with safe_open(manifest.snapshot / metadata.shard, framework="pt", device="cpu") as source:
            value = source.get_tensor(name)
        projection, suffix = name.rsplit(".", 2)[-2:]
        if suffix == "mcg":
            if value.item() != MCG_MARKER:
                raise ValueError(f"{name}: invalid MCG marker")
            continue
        dimension = None
        if projection in ("gate_proj", "up_proj"):
            dimension = 1 if suffix == "trellis" else 0 if suffix == "svh" else None
        elif projection == "down_proj":
            dimension = 0 if suffix in ("trellis", "suh") else None
        if dimension is not None:
            size = value.shape[dimension] // 2
            value = value.narrow(dimension, tp_rank * size, size)
        tensor = value.contiguous().to(device)
        # Trellis integers contain numeric codes, not CUDA pointers. The pointer
        # tables built below deliberately remain opaque to package reachability.
        record_numeric_tensor(tensor)
        tensors[f"{projection}.{suffix}"] = tensor
    return tensors


class CooperativeCall:
    """Fixed caller-owned storage for the existing native launch ABI."""

    def __init__(self, library: Path, weights, rows: int, device):
        import torch
        from ignition.preparation.tensor_metadata import record_numeric_tensor

        self.library = C.CDLL(str(library))
        self.library.glm53_coop_abi.restype = C.c_int
        if self.library.glm53_coop_abi() != 1:
            raise ValueError("unexpected glm53_coop_abi")
        self.library.glm53_coop_info.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_int)]
        self.library.glm53_coop_info.restype = C.c_int
        info = (C.c_int * 18)()
        if self.library.glm53_coop_info(BITS, GEOMETRY, info):
            raise RuntimeError("glm53_coop_info failed")
        if info[16] != COUNTER_ELEMENTS or info[17] != 344 or min(info[4], info[9], info[14]) < 1:
            raise ValueError("cooperative native layout or residency differs from the validated ABI")
        self.launch = self.library.glm53_coop_launch
        self.launch.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_int, C.c_int,
                                C.c_float, C.c_int, C.c_int, C.c_void_p]
        self.launch.restype = C.c_int
        self.source = torch.linspace(-.5, .5, rows * HIDDEN, device=device, dtype=torch.float32).half().view(rows, HIDDEN)
        self.ids = torch.arange(TOPK, device=device, dtype=torch.int64).repeat(rows, 1)
        self.route_weights = torch.full((rows, TOPK), 1 / TOPK, device=device, dtype=torch.float16)
        self.tables = [torch.full((EXPERT_SLOTS,), weights[f"{projection}.{suffix}"].data_ptr(),
                                 device=device, dtype=torch.int64)
                       for projection in ("gate_proj", "up_proj", "down_proj")
                       for suffix in ("trellis", "suh", "svh")]
        self.scratch = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in (
            ((SLOTS_MAX, HIDDEN), torch.float16), ((SLOTS_MAX, HIDDEN), torch.float16),
            ((SLOTS_MAX, INTERMEDIATE), torch.float16), ((SLOTS_MAX, INTERMEDIATE), torch.float16),
            ((SLOTS_MAX, INTERMEDIATE), torch.float16), ((SLOTS_MAX, HIDDEN), torch.float32))]
        self.counters = torch.zeros(COUNTER_ELEMENTS, dtype=torch.int32, device=device)
        self.output = torch.empty((rows, HIDDEN), dtype=torch.float32, device=device)
        record_numeric_tensor(self.ids)
        self.weights, self.rows, self.device = weights, rows, device
        tensors = (self.source, self.ids, self.route_weights, *self.tables,
                   *self.scratch, self.counters, self.output)
        self.pointers = (C.c_void_p * 20)(*[tensor.data_ptr() for tensor in tensors])

    def __call__(self):
        import torch

        status = self.launch(self.pointers, BITS, self.rows, EXPERT_SLOTS, 10., GEOMETRY, 0,
                             C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream))
        if status:
            raise RuntimeError(f"cooperative EXL3 launch failed with CUDA status {status}")


def export_probe(args):
    manifest, contract = checkpoint_plan(args.checkpoint, args.layer, args.expert)
    validate_library(args.library)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise ValueError("assign CUDA_VISIBLE_DEVICES explicitly before running the GPU probe")

    import torch
    from helios.tests.fixture_frontend import server_config
    from helios.tests.signing import publisher
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.preparation.runtime_package import export_model

    with ModuleImages() as images:
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability(0) != (12, 1):
            raise ValueError("the validated cooperative probe requires SM121")
        args.output.mkdir(parents=True, exist_ok=False)
        with PackageStorage(device=0) as storage:
            device = torch.device("cuda", 0)
            weights = load_expert(manifest, contract, args.tp_rank, device)
            call = CooperativeCall(args.library, weights, args.rows, device)
            call.output.fill_(math.nan)
            call()
            torch.cuda.synchronize()
            if not torch.isfinite(call.output).all().item() or not call.output.count_nonzero().item():
                raise RuntimeError("EXL3 reference must be finite and nonzero")
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            try:
                with torch.cuda.graph(graph, pool=storage.graph_pool.id):
                    call()
                graph.instantiate()
                # Native loading restores these disposable counters to zero.
                # Compare against that same initial state, not warmup leftovers.
                call.counters.zero_()
                call.output.fill_(math.nan)
                graph.replay()
                torch.cuda.synchronize()
                if not torch.isfinite(call.output).all().item() or not call.output.count_nonzero().item():
                    raise RuntimeError("captured EXL3 reference must be finite and nonzero")
                expected = call.output.view(torch.uint8).cpu().numpy().tobytes()
                keys = ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed")
                before = tuple(torch.cuda.memory_stats()[key] for key in keys)
                for _ in range(7):
                    graph.replay()
                torch.cuda.synchronize()
                after = tuple(torch.cuda.memory_stats()[key] for key in keys)
                if before != after:
                    raise RuntimeError("captured EXL3 replay allocated or freed CUDA storage")
                if call.output.view(torch.uint8).cpu().numpy().tobytes() != expected:
                    raise RuntimeError("EXL3 repeated capture replay is not byte-stable; native comparison would be invalid")
                buffers = {"frame_index": torch.zeros(1, device=device, dtype=torch.int32),
                           "submission_sequence": torch.zeros(1, device=device, dtype=torch.int64),
                           "completion_records": torch.zeros((2, 1, 8), device=device, dtype=torch.int64),
                           "source": call.source, "output": call.output}
                disposable = (*call.scratch, call.counters, call.output)
                prepared = PreparedExecution(device=0, graphs={"decode": graph}, buffers=buffers,
                    signatures={"decode": GraphSignature("decode", 1, 1, ("source",))}, max_active=1,
                    resources=(storage, call),
                    export_zero_ranges=tuple((tensor.data_ptr(), tensor.nbytes) for tensor in disposable))
                prepared.server_config = server_config(args.output)
                prepared.package_data = None
                # Keep the allocator, library, weights and graph alive through publication.
                with torch.cuda.stream(torch.cuda.default_stream()):
                    package = export_model(prepared, images, storage, args.output / "package", options={},
                                           publisher=publisher())
                (args.output / "expected.json").write_text(json.dumps({
                    "reference": package._reference(call.output.data_ptr(), call.output.nbytes),
                    "bytes": list(expected),
                    "addresses": {allocation.name: allocation.address for allocation in package.allocations},
                    "zero_reference": package._reference(call.counters.data_ptr(), call.counters.nbytes),
                    "zero_bytes": call.counters.nbytes,
                }))
            finally:
                torch.cuda.synchronize()
                graph.reset()
    print("EXL3 subset exported; native package reload/replay is still required", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("action", choices=("export", "run"))
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--library", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--native-executable", type=Path)
    result.add_argument("--layer", type=int, default=3)
    result.add_argument("--expert", type=int, default=0)
    result.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    result.add_argument("--rows", type=int, choices=range(1, MAX_ROWS + 1), default=1)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        tensor_contract(args.layer, args.expert)
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        if args.action == "export":
            export_probe(args)
            return
        if args.native_executable is None or not args.native_executable.is_file():
            raise ValueError("run requires --native-executable built from helios test_fixed_runtime_package")
        arguments = [sys.executable, "-m", "qualify.exl3_package", "export"]
        for name in ("checkpoint", "library", "output", "layer", "expert", "tp_rank", "rows"):
            arguments.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
        # The exporting child must exit before the independent Rust context reserves
        # the recorded addresses. The orchestrator itself never imports Torch.
        subprocess.run(arguments, check=True)
        subprocess.run([str(args.native_executable.resolve()), str(args.output.resolve())], check=True)
        print("EXL3 subset capture/export/native reload/replay passed; full GLM remains unqualified", flush=True)
    except (OSError, ValueError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
