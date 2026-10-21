"""Padded EXL3 graphs retain the actual-row cooperative/thin/E3 arithmetic.

Two real experts are alternated across a serial 42-call shared-workspace trace.
This tests capacity transitions and scratch ownership; it does not qualify all
288 distinct expert weights, the complete decoder, collectives or HTTP.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("b12x")

from ignition.components.exl3 import Exl3Experts, PreparedExl3Experts
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.glm53.weights import GlmWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires explicitly assigned idle SM121 and original deployment libraries")


class PaddedExperts:
    def __init__(self, source, experts, ids, weights, output, live, calls):
        self.ids, self.weights, self.live = ids, weights, live
        self.masked_ids, self.masked_weights = torch.empty_like(ids), torch.empty_like(weights)
        self.rows = torch.arange(len(source), device=source.device)
        self.operations = tuple(PreparedExl3Experts(source, experts[index % len(experts)],
            self.masked_ids, self.masked_weights, output, num_valid_tokens=live) for index in range(calls))

    def __call__(self):
        valid = self.rows[:, None] < self.live
        self.masked_ids.copy_(torch.where(valid, self.ids, -1))
        self.masked_weights.copy_(torch.where(valid, self.weights, 0.))
        for operation in self.operations:
            operation()


@pytest.mark.parametrize("calls", [1, 42])
@torch.no_grad()
def test_compiled_inactive_suffix_and_serial_layer_dispatch(calls):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    for name in ("FLARE_EXL3_CHECKPOINT", "FLARE_EXL3_LIBRARY", "FLARE_EXL3_PREFILL_LIBRARY"):
        assert os.environ.get(name), name
    device, capacity = torch.device("cuda:0"), 128
    prefix = "model.language_model.layers.3.mlp.experts"
    with GlmWeightLoader(inspect_indexed_checkpoint(Path(os.environ["FLARE_EXL3_CHECKPOINT"])), device,
            shard=TensorParallelShard(1, 2), prefixes=tuple(f"{prefix}.{expert}." for expert in (0, 287)),
            exl3_library=Path(os.environ["FLARE_EXL3_LIBRARY"]),
            exl3_prefill_library=Path(os.environ["FLARE_EXL3_PREFILL_LIBRARY"]),
            exl3_prefill_capacity=capacity) as loader:
        experts = loader.load_experts(3, expert_ids=(0, 287))
    swapped = Exl3Experts.from_tensors(tuple(reversed(experts.tensors)), experts.workspace)
    source = torch.empty((capacity, 4096), dtype=torch.bfloat16, device=device)
    source.normal_(std=.5)
    original_source = source.clone()
    ids = torch.full((capacity, 8), 2, dtype=torch.int32, device=device)
    weights = torch.zeros((capacity, 8), dtype=torch.float32, device=device)
    ids[:, 0] = 0
    ids[::2, 1] = 1
    weights[:, 0], weights[::2, 1] = .6, .4
    output, live = torch.empty_like(source), torch.zeros(1, dtype=torch.int32, device=device)
    forward = PaddedExperts(source, (experts, swapped), ids, weights, output, live, calls)
    compiled = compile_forward(forward)
    compiled()
    graph, worker = torch.cuda.CUDAGraph(), None
    try:
        with torch.cuda.graph(graph):
            compiled()
        worker = PreparedGraphBundle(device=0, graphs={"experts": graph}, buffers=dict(source=source,
            output=output, live=live, ids=ids, weights=weights), resources=(forward, compiled, experts, swapped)).start_worker()
        for count in (17, 65, 0, 32, 33, 1, 128, 17):
            live.fill_(count)
            source.copy_(original_source)
            source[count:].fill_(float("nan"))
            expected = torch.zeros_like(output)
            if count:
                actual_rows = PreparedExl3Experts(source[:count], (experts, swapped)[(calls - 1) % 2],
                    ids[:count], weights[:count], expected[:count])
                actual_rows()
            # Poison scratch left by the independent reference and previous graph variant.
            output.fill_(float("nan"))
            torch.cuda.synchronize()
            keys = ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed")
            before = tuple(torch.cuda.memory_stats()[key] for key in keys)
            ticket = worker.submit(0, repetitions=3)
            completed = worker.poll(timeout_ms=120000)
            assert completed is not None and completed.ticket == ticket and completed.error is None
            after = tuple(torch.cuda.memory_stats()[key] for key in keys)
            assert before == after
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            assert (forward.masked_ids[count:] == -1).all()
            assert not forward.masked_weights[count:].count_nonzero()
    finally:
        if worker is not None:
            worker.close()
        graph.reset()
        experts.workspace.close()
