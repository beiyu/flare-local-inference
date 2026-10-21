"""Real-weight, captured-native component experiments on the two Spark GPUs.

Run communication on both ranks together, then branches independently. This is
qualification code, not a serving path or a replacement full-model correctness
gate. Outputs are immutable numeric reports; checkpoint precision is unchanged.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist

from ignition.components.collectives import PreparedAllGather
from ignition.components.linear import UnquantizedProjection
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import CollectivePreparation, create_roce_runtime
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53.weights import GlmWeightLoader


def save(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def session(name, coordinator=None):
    return RecordedPreparationSession(device=torch.device("cuda:0"), checkpoint="glm53-exl3-dflash2-original",
        workload_schema="glm-optimization-components-v1", numerical_policy="default",
        coordinator=coordinator, compile_workers=2)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def timeline(operations, path):
    """Attribute actual emitted GPU kernels through CUPTI launch correlations."""
    from torch.profiler import ProfilerActivity, profile, record_function
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as capture:
        for name, function in operations:
            barrier()
            with record_function(name):
                function()
            torch.cuda.synchronize()
    capture.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    labels = {name for name, _ in operations}
    ranges = [e for e in events if e.get("name") in labels and "dur" in e]
    launches = {e.get("args", {}).get("correlation"): e for e in events
                if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
    groups = {name: [] for name in labels}
    for event in events:
        if event.get("cat") != "kernel" or "dur" not in event:
            continue
        launch = launches.get(event.get("args", {}).get("correlation"))
        if launch is None:
            continue
        for scope in ranges:
            if scope["ts"] <= launch["ts"] <= scope["ts"] + scope["dur"]:
                groups[scope["name"]].append(event)
                break
    return {"trace": str(path), "operations": {name: {
        "kernel_count": len(kernels), "kernel_sum_us": sum(k["dur"] for k in kernels),
        "gpu_span_us": (max(k["ts"] + k["dur"] for k in kernels) - min(k["ts"] for k in kernels)) if kernels else None,
        "kernel_names": sorted({k["name"] for k in kernels}),
    } for name, kernels in groups.items()}}


def measure(function, name, buffers, *, repetitions=32):
    """GPU events inside the replayed graph exclude Python submission gaps."""
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    barrier()
    with torch.cuda.graph(graph):
        begin.record()
        function()
        end.record()
    barrier()
    bundle = PreparedGraphBundle(device=0, graphs={name: graph}, buffers=buffers,
        resources=(function, begin, end))
    worker = bundle.start_worker()
    times = []
    try:
        for _ in range(7):
            barrier()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(0, repetitions=repetitions)
            result = worker.poll(timeout_ms=120000)
            assert result is not None and result.ticket == ticket and result.error is None
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for counter in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[counter] == after[counter], (name, counter)
            elapsed = begin.elapsed_time(end) * 1000
            if dist.is_initialized():
                ranks = [None] * dist.get_world_size()
                dist.all_gather_object(ranks, elapsed)
                elapsed = max(ranks)
            times.append(elapsed)
    finally:
        worker.close()
        graph.reset()
    return {"name": name, "median_us": statistics.median(times), "min_us": min(times), "samples_us": times,
            "native_replay_allocation_stable": True,
            "rank_span": "max rank GPU interval" if dist.is_initialized() else "local rank GPU interval"}


class Candidates:
    def __init__(self, local, runtime, coordinator, preparation, *, rank, compact):
        self.local, self.compact, self.rank = local, compact, rank
        rows, columns = local.shape
        width = 48 if compact else columns
        self.send = torch.empty((rows, width), dtype=torch.bfloat16, device=local.device)
        self.received = torch.empty((rows, width * 2), dtype=local.dtype, device=local.device)
        self.scores = torch.empty((rows, 16), dtype=local.dtype, device=local.device)
        self.ids = torch.empty((rows, 16), dtype=torch.int64, device=local.device)
        self.gather = PreparedAllGather(self.send if compact else local, self.received, runtime=runtime,
            session=preparation, coordinator=coordinator, name="compact" if compact else "full")

    def __call__(self):
        if self.compact:
            values, ids = self.local.topk(16, dim=-1)
            self.send[:, :16].copy_(values)
            self.send[:, 16:].copy_((ids + self.rank * self.local.shape[1]).int().view(torch.bfloat16))
            gathered = self.gather(self.send).view(self.local.shape[0], 2, 48)
            values = gathered[:, :, :16].reshape(-1, 32)
            ids = gathered[:, :, 16:].contiguous().view(torch.int32).reshape(-1, 32)
            scores, positions = values.topk(16, dim=-1)
            self.ids.copy_(ids.gather(1, positions).long())
            self.scores.copy_(scores)
        else:
            values, ids = self.gather(self.local).topk(16, dim=-1)
            self.ids.copy_(ids)
            self.scores.copy_(values)


@torch.no_grad()
def communication(args):
    dist.init_process_group("gloo", init_method=args.rendezvous, rank=args.rank, world_size=2,
        timeout=timedelta(minutes=10))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group)
    runtime = create_roce_runtime(group, torch.device("cuda:0"), max_tokens=8, pipe_tokens=256,
        hidden_size=4096, gather_bytes=8 * 77440 * 2)
    manifest = inspect_indexed_checkpoint(args.checkpoint)
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(args.rank, 2), prefixes=("lm_head.",),
            exl3_library=args.exl3_library) as loader:
        weight = loader.load_tensor("lm_head.weight", extent=slice(args.rank * 77440, (args.rank + 1) * 77440))
    hidden = torch.empty((8, 4096), dtype=torch.bfloat16, device="cuda:0")
    local = torch.empty((8, 77440), dtype=torch.bfloat16, device="cuda:0")
    report = {"rank": args.rank, "kind": "communication", "timings": [], "correctness": [],
              "full_payload_bytes_per_rank": local.numel() * 2, "compact_payload_bytes_per_rank": 8 * 48 * 2}
    with session("communication", coordinator) as preparation:
        projection = UnquantizedProjection(hidden, weight, local, session=preparation, name="head")
        full = Candidates(local, runtime, coordinator, preparation, rank=args.rank, compact=False)
        compact = Candidates(local, runtime, coordinator, preparation, rank=args.rank, compact=True)
        preparation.tune()
        generator = torch.Generator(device="cuda:0").manual_seed(531)
        for case in ("real0", "real1", "real2", "constant-ties", "cutoff-ties", "shard-boundary-ties"):
            if case.startswith("real"):
                hidden.copy_(torch.randn(hidden.shape, device=hidden.device, dtype=hidden.dtype, generator=generator))
                projection(hidden)
            elif case == "constant-ties":
                local.fill_(1)
            else:
                local.fill_(-10)
                local[:, :32].fill_(1)
                if case == "cutoff-ties":
                    local[:, :7].copy_(torch.arange(9, 16, device=local.device, dtype=local.dtype))
            full()
            compact()
            torch.cuda.synchronize()
            original = torch.full_like(full.received, -float("inf"))
            merged = torch.full_like(original, -float("inf"))
            original.scatter_(1, full.ids, full.scores)
            merged.scatter_(1, compact.ids, compact.scores)
            report["correctness"].append({"case": case, "scores_equal": torch.equal(full.scores, compact.scores),
                "ordered_ids_equal": torch.equal(full.ids, compact.ids),
                "candidate_support_and_unary_equal": torch.equal(original, merged),
                "different_ids": (full.ids != compact.ids).sum().item()})
        hidden.copy_(torch.randn(hidden.shape, device=hidden.device, dtype=hidden.dtype, generator=generator))
        projection(hidden)
        # The exact same scalar/native operations are first traced and compiled.
        compiled_full, compiled_compact = compile_forward(full), compile_forward(compact)
        buffers = dict(hidden=hidden, local=local, full=full.received, compact=compact.received)
        operations = [("head-projection", lambda: projection(hidden)), ("full-gather", lambda: full.gather(local)),
            ("compact-gather-only", lambda: compact.gather(compact.send)),
            ("full-candidates-compiled", compiled_full), ("compact-candidates-compiled", compiled_compact),
            ("head-full-compiled", lambda: (projection(hidden), compiled_full())),
            ("head-compact-compiled", lambda: (projection(hidden), compiled_compact()))]
        for name, function in operations:
            report["timings"].append(measure(function, name, buffers))
            print(json.dumps({"rank": args.rank, **report["timings"][-1]}), flush=True)
        try:
            report["kernel_timeline"] = timeline(operations, args.output.with_suffix(".trace.json"))
        except Exception as error:
            report["kernel_timeline_error"] = str(error)
    save(args.output, report)
    dist.destroy_process_group()


@torch.no_grad()
def branches(args):
    from ignition.components.expert_overlap import ConcurrentSharedExpert
    from ignition.recipes.glm53.kda import KDAScratch
    from ignition.recipes.glm53.model import create_model_tree
    from ignition.recipes.glm53.moe import MoeScratch, PreparedGlmMoe

    manifest = inspect_indexed_checkpoint(args.checkpoint)
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(args.rank, 2),
            prefixes=("model.language_model.layers.0.self_attn.", "model.language_model.layers.3.mlp."),
            exl3_library=args.exl3_library) as loader:
        kda_weights = loader.load_attention(0)
        moe_weights = loader.load_mlp(3)
    tree = create_model_tree(args.checkpoint)
    report = {"rank": args.rank, "kind": "branches", "results": [],
              "qualification": "Real-weight component upper bounds; not full-forward/per-service speedup claims."}
    stream = torch.cuda.Stream()
    for rows in (1, 4, 8):
        source = torch.randn((rows, 4096), dtype=torch.bfloat16, device="cuda:0")
        scratch = KDAScratch.allocate(rows, 32, "cuda:0")
        with session("branches") as preparation:
            projections = {key: UnquantizedProjection(x, kda_weights[key], y, session=preparation, name=key)
                for key, x, y in (("input_proj", source, scratch.projected_input),
                    ("f_b_proj", scratch.forget_input, scratch.raw_g),
                    ("g_a_proj", source, scratch.gate_input), ("g_b_proj", scratch.gate_input, scratch.z))}
            arena = ScratchArena(torch.device("cuda:0"))
            preparation.register_scratch(arena)
            moe_scratch = MoeScratch.allocate(rows, 2, "cuda:0")
            moe = PreparedGlmMoe(tree.model.language_model.layers[3].mlp, source, moe_weights, moe_scratch,
                session=preparation, name="moe", workspace=arena)
            # EXL3 owns its workspace; routing and the BF16 shared MLP use the
            # arena. The existing overlap component assigns the MLP a disjoint
            # suffix and retains the fork/join dependencies.
            overlap = ConcurrentSharedExpert(moe.shared, parallel_plans=(moe.routing.plan,),
                session=preparation, workspace=arena)
            preparation.tune()

            def kda(parallel=False):
                main = torch.cuda.current_stream()
                if parallel:
                    stream.wait_stream(main)
                    with torch.cuda.stream(stream):
                        projections["g_b_proj"](projections["g_a_proj"](source))
                projections["input_proj"](source)
                if rows > 1:
                    scratch.forget_input.copy_(scratch.projected_input[:, -128:])
                projections["f_b_proj"](scratch.forget_input)
                if parallel:
                    main.wait_stream(stream)
                else:
                    projections["g_b_proj"](projections["g_a_proj"](source))

            def experts(parallel=False, defer=False):
                if not parallel:
                    moe(source)
                    return
                moe.router_projection(source)
                overlap.defer_launch = defer
                overlap.start(source)
                moe.routing(moe_scratch.router_logits)
                routed = moe.experts()
                shared = overlap.finish(source, routed)
                torch.add(routed, shared, out=moe.output)

            for branch, serial, variants, outputs in (
                ("kda", kda, [("parallel", lambda: kda(True))], (scratch.projected_input, scratch.raw_g, scratch.z)),
                ("experts", experts, [("parallel-before", lambda: experts(True)),
                                     ("parallel-after", lambda: experts(True, True))], (moe.output,)),
            ):
                serial()
                torch.cuda.synchronize()
                reference = [x.clone() for x in outputs]
                for name, function in [("serial", serial)] + variants:
                    function()
                    torch.cuda.synchronize()
                    exact = all(torch.equal(x, y) for x, y in zip(outputs, reference))
                    timing = measure(function, f"{branch}-rows{rows}-{name}", dict(source=source))
                    report["results"].append({"branch": branch, "rows": rows, "variant": name,
                        "output_exact": exact,
                        "native_output_exact": all(torch.equal(x, y) for x, y in zip(outputs, reference)),
                        **timing})
                    print(json.dumps({"rank": args.rank, **report["results"][-1]}), flush=True)
                try:
                    trace = args.output.with_name(f"{args.output.stem}-{branch}-{rows}.trace.json")
                    report.setdefault("kernel_timelines", []).append(timeline([("serial", serial)] + variants, trace))
                except Exception as error:
                    report.setdefault("kernel_timeline_errors", []).append(str(error))
    save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("communication", "branches"))
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--rendezvous", default="tcp://10.77.13.1:29671")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--exl3-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES explicitly"
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.cuda.set_device(0)
    torch.manual_seed(531 + args.rank)
    {"communication": communication, "branches": branches}[args.kind](args)


if __name__ == "__main__":
    main()
