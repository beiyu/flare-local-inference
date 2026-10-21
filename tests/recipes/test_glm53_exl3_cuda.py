"""Real EXL3 TP2 slices through complete-forward compilation and native replay.

This establishes the cooperative composition against its own per-row native
arithmetic, including two distinct checkpoint experts. It does not establish
agreement with the deployment's separate E3 fat-expert prefill arithmetic.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("b12x")

from ignition.components.exl3 import PreparedCooperativeExl3Experts
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.forward_metadata import dumps, loads
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.glm53.weights import GlmWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned idle SM121 GPU and the deployment binary")


@pytest.mark.parametrize("rows", [1, 32, 33, 65])
@torch.no_grad()
def test_distinct_checkpoint_experts_compiled_native_replay(rows):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA_VISIBLE_DEVICES explicitly"
    for name in ("FLARE_EXL3_CHECKPOINT", "FLARE_EXL3_LIBRARY"):
        assert os.environ.get(name), f"set {name} to the actual EXL3 input"
    manifest = inspect_indexed_checkpoint(Path(os.environ["FLARE_EXL3_CHECKPOINT"]))
    prefix = "model.language_model.layers.3.mlp.experts"
    device = torch.device("cuda:0")
    with GlmWeightLoader(manifest, device, shard=TensorParallelShard(0, 2),
            prefixes=tuple(f"{prefix}.{expert}." for expert in (0, 287)),
            exl3_library=Path(os.environ["FLARE_EXL3_LIBRARY"])) as loader:
        experts = loader.load_experts(3, expert_ids=(0, 287))
    source = torch.empty((rows, 4096), dtype=torch.bfloat16, device=device)
    ids = (torch.arange(rows * 8, device=device).view(rows, 8) % 2).to(torch.int32)
    weights = torch.linspace(.01, .2, rows * 8, device=device).view(rows, 8)
    output = torch.empty_like(source)
    prepared = PreparedCooperativeExl3Experts(source, experts, ids, weights, output)
    # Metadata serialization must strip the CDLL and CUDA resources while preserving the trace.
    traced = loads(dumps(prepared))
    assert traced.slices == prepared.slices
    assert traced.experts.tables[0].device.type == "cuda"
    compiled = compile_forward(prepared)
    source.normal_(std=.1)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    worker = None
    try:
        with torch.cuda.graph(graph):
            compiled()
        bundle = PreparedGraphBundle(device=0, graphs={"experts": graph},
            buffers=dict(source=source, output=output, ids=ids, weights=weights),
            resources=(prepared, compiled, experts))
        worker = bundle.start_worker()
        for scale in (.1, .5):
            source.normal_(std=scale)
            expected = torch.empty_like(output)
            # Independent physical row counts expose a stale shared-scratch or cross-slice state bug.
            for row in range(rows):
                reference = PreparedCooperativeExl3Experts(source[row:row + 1], experts, ids[row:row + 1],
                    weights[row:row + 1], expected[row:row + 1])
                reference()
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(0, repetitions=3)
            result = worker.poll(timeout_ms=120000)
            assert result is not None and result.ticket == ticket and result.error is None
            after = torch.cuda.memory_stats()
            for counter in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                assert before[counter] == after[counter]
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        if worker is not None:
            worker.close()
        graph.reset()
