"""GLM encoder compilation, native replay, publication and scratch lifetime checks.

These tests require real checkpoint weights and an explicitly assigned GPU.
Matching pooled/unpooled execution does not establish vLLM numerical parity.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device and actual GLM checkpoint")


@pytest.mark.parametrize("capacity", [64, 256, 8192])
@torch.no_grad()
def test_glm_encoder_compiles_and_native_replays_after_poisoning(capacity):
    from b12x.preparation import PreparationSession
    from tests.components.completion_support import completion_outputs
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.glm_vision_kernels import OriginalVisionKernels
    from ignition.components.vision_state import PrefillVisionInputs, VisionState
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.glm53.vision import IMAGE_TOKEN, PreparedVision, VisionInputs, VisionScratch
    from ignition.recipes.glm53.vision_state import PreparedVisionGeneration
    from ignition.recipes.glm53.vision_weights import VisionWeights

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    snapshot = Path(os.environ["FLARE_GLM_CHECKPOINT"])
    device = torch.device("cuda:0")
    weights = VisionWeights.load(inspect_indexed_checkpoint(snapshot), device)
    kernels = OriginalVisionKernels(Path(os.environ["FLARE_GLM_VISION_KERNELS"]), device)
    pool, arena = GraphBufferPool(device, chunk_bytes=2 << 20), ScratchArena(device)
    inputs = VisionInputs.allocate(device)
    inputs.vision_slot.fill_(1)
    inputs.vision_generation.fill_(7)
    requests = DeviceRequestState.allocate(device, vocab_size=154880, capacity=2)
    requests.generation[1], requests.status[1] = 7, 1
    batch = SimpleNamespace(token_ids=torch.zeros(4, dtype=torch.int64, device=device),
        row_slot_ids=torch.ones(1, dtype=torch.int32, device=device), row_generations=inputs.vision_generation)
    records = torch.zeros((2, 2, 24), dtype=torch.int64, device=device)
    frame, sequence = torch.zeros(1, dtype=torch.int32, device=device), torch.ones(1, dtype=torch.int64, device=device)
    graphs, owners, states = {}, [], []
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        for pooled in (False, True):
            buffers = pool.cursor() if pooled else None
            scratch = VisionScratch.allocate(capacity, device, buffers=buffers)
            encoder = PreparedVision(weights, scratch, inputs, kernels=kernels, session=session, workspace=arena,
                name=f"glm-vision-{pooled}", buffers=buffers)
            state = VisionState(batch, PrefillVisionInputs.allocate(4, device), requests,
                image_token=IMAGE_TOKEN, image_slots=1, features=2048, hidden=4096)
            state.feature_slot.fill_(0)
            operation = PreparedVisionGeneration(encoder, state, completion_outputs(records), frame, sequence, buffers=buffers)
            states.append(state)
            owners.append(operation)
        arena.finalize()
        calls = [compile_forward(operation) for operation in owners]
        for call in calls:
            call()
        session.freeze()
        for index, call in enumerate(calls):
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                call()
            graphs[str(index)] = graph
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={},
            resources=(owners, calls, states, weights, inputs, requests, pool, arena, session))
        worker = bundle.start_worker()
        try:
            for count, height in ((64, 8), (capacity, {64: 8, 256: 16, 8192: 64}[capacity]), (80, 10)):
                if count > capacity:
                    continue
                width = count // height
                positions = torch.stack(torch.meshgrid(torch.arange(height, device=device),
                    torch.arange(width, device=device), indexing="ij"), -1)
                positions = positions.reshape(height // 2, 2, width // 2, 2, 2).permute(0, 2, 1, 3, 4).reshape(count, 2)
                inputs.vision_position_ids[:count].copy_(positions)
                inputs.vision_cu_seqlens[1] = count
                inputs.vision_num_patches.fill_(count)
                inputs.vision_num_features.fill_(count // 4)
                inputs.vision_pixel_values[:count].copy_(torch.randn((count, 1176), device=device,
                    generator=torch.Generator(device=device).manual_seed(count)))
                for chunks in pool.chunks.values():
                    for tensor in chunks:
                        tensor.fill_(float("nan") if tensor.is_floating_point() else -37)
                requests.status[1], requests.error_code[1] = 1, 0
                for state in states:
                    state.arena.fill_(float("nan"))
                    state.count.zero_()
                    state.generation.zero_()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                for index in (0, 1):
                    ticket = worker.submit(index)
                    completion = worker.poll(timeout_ms=30000)
                    assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                expected, actual = (state.arena[0, :count // 4] for state in states)
                assert torch.isfinite(expected).all()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for state in states:
                    assert state.count.tolist() == [0, count // 4]
                    assert state.generation.tolist() == [0, 7]
                    assert torch.isnan(state.arena[0, count // 4:]).all()
        finally:
            worker.close()
            for graph in graphs.values():
                graph.reset()
