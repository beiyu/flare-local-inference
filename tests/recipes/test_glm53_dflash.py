"""DFlash2 geometry and exact-prefix storage remain distinct from integrated GLM MTP."""

import json

import pytest
import torch

from ignition.recipes.glm53.dflash_weights import DraftConfig
from ignition.recipes.glm53.pool import ContinuationPool, continuation_layout


def checkpoint_config():
    return dict(architectures=["DFlash2DraftModel"], hidden_size=4096, intermediate_size=12288,
        num_hidden_layers=5, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        num_target_layers=45, vocab_size=154880, hidden_act="silu", attention_bias=False,
        is_causal=False, tie_word_embeddings=False, dtype="bfloat16", layer_types=["sliding_attention"] * 5,
        sliding_window=2048, rms_norm_eps=1e-5, rope_parameters=dict(rope_theta=10000., rope_type="default"),
        dflash_config=dict(block_size=8, conv_group_size=16, conv_kernel_size=2, mask_token_id=154856,
            selector_rank=256, selector_top_k=16, target_layer_ids=[5, 14, 24, 33, 42]))


@pytest.fixture
def draft_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config()))
    return DraftConfig.load(tmp_path)


@pytest.mark.parametrize("field,value", [("is_causal", True), ("num_target_layers", 46),
    ("vocab_size", 151936), ("dtype", "float16"), ("quantization_config", {"quant_method": "fp8"}),
    ("sliding_window", 4096)])
def test_incompatible_draft_rejected_before_loading(tmp_path, field, value):
    raw = checkpoint_config()
    raw[field] = value
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        DraftConfig.load(tmp_path)


@pytest.mark.parametrize("world", [1, 2, 4])
def test_draft_windows_snapshot_restore_with_complete_target_state(draft_config, world):
    pool = ContinuationPool.allocate(("linear_attention", "indexed_attention"), world, 3, 1,
        "cpu", dflash2=draft_config)
    assert pool.draft is None  # The checkpoint-integrated MLA MTP is not substituted for DFlash2.
    assert pool.decoder.layers[0].convolution.shape[-1] == 10
    assert pool.decoder.layers[1].index.tail.shape[-2] == 12
    assert not tuple(pool.draft_pages())
    for i in range(5):
        for kind in ("key", "value"):
            window = pool.region(f"dflash2.{i}.{kind}")
            assert window.shape == (3, 2048, 8 // world, 128)
            assert window.stride(0) * window.element_size() == pool.layout.row_bytes
            window[1, 2047].fill_(i + 1)
            window[1, 0].fill_(i + 9)
    saved = pool.backing[1].clone()
    pool.backing[2].copy_(pool.backing[1])
    pool.backing[1].zero_()
    pool.backing[1].copy_(pool.backing[2])
    assert torch.equal(pool.backing[1], saved)
    assert not pool.backing[0].count_nonzero()
    pool.clear_for_preparation()
    assert not pool.backing.count_nonzero()


def test_declared_complete_model_dflash_state_cost(draft_config):
    kinds = tuple("indexed_attention" if i % 4 == 3 else "linear_attention" for i in range(45))
    layout = continuation_layout(kinds, 2, dflash2=draft_config)
    assert layout.row_bytes == 102286336



@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("width", [1, 8])
def test_target_context_shards_and_draft_windows_use_their_own_positions(rank, width):
    from pathlib import Path
    from types import SimpleNamespace
    from ignition.components.paged_batch import Metadata
    from ignition.components.recurrent import DeltaMetadata
    from ignition.recipes.glm53.batch import BatchInputs
    from ignition.recipes.glm53.bootstrap import PreparationConfig
    from ignition.recipes.glm53.dcp import ContextShards
    from ignition.recipes.glm53.dflash_speculative import PreparedDFlashRound
    from ignition.recipes.glm53.state import DecoderMetadata

    config = PreparationConfig(mtp=True, dflash_checkpoint=Path("dflash2"), context_parallel=True,
        speculative_widths=tuple(range(1, 9)))
    assert config.page_tokens == 512
    inputs = BatchInputs.allocate(128, 2, "cpu", capacity=2)
    inputs.page_tables.copy_(torch.tensor([[10, 11], [20, 21]]))
    inputs.request_state_rows[:, 0].copy_(torch.tensor([1, 2]))
    inputs.live_rows.fill_(2)
    op = PreparedDFlashRound.__new__(PreparedDFlashRound)
    op.owner = SimpleNamespace(inputs=inputs, requests=SimpleNamespace(capacity=2), shards=ContextShards.on(rank, 2, "cpu"))
    op.batch, op.width = 2, width
    m = DecoderMetadata.allocate(2 * width, 2, 2, "cpu")
    op.metadata = DecoderMetadata(m.token_ids, torch.zeros(2 * width, dtype=torch.int64),
        DeltaMetadata.allocate(2, "cpu", verification_columns=8 if width > 1 else 1), m.indexed)
    op.active = torch.tensor([True, True])
    op.before = torch.tensor([511, 1023])
    op.offsets, op.columns = torch.arange(2 * width), torch.arange(width)
    op.state_columns = torch.arange(8 if width > 1 else 1)
    op.candidates = torch.arange(2 * width).view(2, width)
    op.target_metadata(torch.tensor([0, 1]), torch.tensor([width, 1], dtype=torch.int32))
    expected_first = ([-1, 2816, 2817, 2818, 2819, -1, -1, -1] if rank == 0 else
        [2815, -1, -1, -1, -1, 2816, 2817, 2818])
    expected = expected_first[:width] + ([-1] if rank == 0 else [5631])
    assert op.metadata.indexed.token_slots[:width + 1].tolist() == expected
    assert op.metadata.recurrent.state_indices[:, 0].tolist() == [1, 2]
    assert not op.metadata.recurrent.state_indices[:, 1:].count_nonzero()
    draft = Metadata.allocate(16, 2, 2, "cpu", samples=16)
    draft.set(op.before, torch.tensor([8, 8]), torch.tensor([1, 2]), inputs.page_tables,
        torch.zeros((2, 8), dtype=torch.int64), inputs.live_rows)
    assert draft.positions.tolist() == list(range(511, 519)) + list(range(1023, 1031))


def test_selector_realizes_only_the_chosen_predecessor_path():
    from ignition.recipes.glm53.dflash import CandidateSelector

    generator = torch.Generator().manual_seed(1923)
    op = CandidateSelector.__new__(CandidateSelector)
    op.weights = {name: torch.randn((32, 256), generator=generator).bfloat16()
                  for name in ("predecessor", "successor")}
    op.hidden = torch.randn((2, 8, 256), generator=generator).bfloat16()
    op.candidates = torch.arange(16)[None, None].expand(2, 8, 16).clone()
    op.unary = torch.randn((2, 8, 16), generator=generator).bfloat16()
    output = torch.empty((2, 32), dtype=torch.float32)
    previous = torch.tensor([20, 25])
    op(output, previous, 3)
    # Independent full pairwise scoring, followed by choosing the actual predecessor.
    predecessor_ids = torch.tensor([[20, 21], [24, 25]])
    predecessor = op.weights["predecessor"][predecessor_ids]
    successors = op.weights["successor"][op.candidates[:, 3]]
    gate = predecessor * op.hidden[:, 3, None]
    all_edges = op.unary[:, 3, None] + torch.einsum("bpr,bcr->bpc", gate, successors)
    expected = torch.stack((all_edges[0, 0], all_edges[1, 1])).float()
    torch.testing.assert_close(output[:, :16], expected, rtol=0, atol=0)
    assert torch.isneginf(output[:, 16:]).all()
    assert torch.equal(output.argmax(-1), expected.argmax(-1))
    probabilities = output.softmax(-1)
    assert not probabilities[:, 16:].count_nonzero()
    assert not torch.equal(all_edges[:, 0], all_edges[:, 1])
