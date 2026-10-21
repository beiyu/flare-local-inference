"""Parallel GLM DFlash2 proposals through the shared stochastic verifier and KDA commit."""

from types import SimpleNamespace

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.kda_speculation import KDARecords
from ignition.components.paged_batch import Metadata
from ignition.components.recurrent import DeltaMetadata
from ignition.components.speculation import DRAFT_RNG_DOMAIN, SpeculativeRngState
from ignition.components.state import save_hidden
from ignition.preparation.scratch import GraphBufferPool

from .batch import PreparedBatch
from .dcp import ContextShards
from .dflash import PreparedContext, PreparedDraft
from .speculative import PreparedKDACommit, PreparedRound, convolution_regions, delta_layers
from .stack import PreparedDecoder
from .state import DecoderMetadata


class SpeculativePreparation:
    draft_context_lag = 0

    def __init__(self, weights, pool, requests, inputs, scratch, *, session, workspace, completion, frame,
                 sequence, runtime, vocabulary_runtime, coordinator, max_cache_pages, context=None, norm_kernels=None):
        self.weights, self.pool, self.requests, self.inputs = weights, pool, requests, inputs
        self.scratch, self.session, self.workspace = scratch, session, workspace
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.max_cache_pages, self.context, self.norm_kernels = max_cache_pages, context, norm_kernels
        self.shards = ContextShards() if context is None else context.shards
        device = pool.backing.device
        self.buffers, self.layer_buffers, self.sampling_buffers = (GraphBufferPool(device) for _ in range(3))
        self.rng = SpeculativeRngState.allocate(device, capacity=requests.capacity)
        self.draft_generation, self.draft_offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        layers = delta_layers(pool)
        recurrent = pool.decoder.layers[layers[0]].recurrent
        self.kda_records = KDARecords.allocate({i: weights.layers[i].attention for i in layers},
            recurrent.shape[0], recurrent.shape[1], device)


class PreparedPrefill:
    def __init__(self, owner, generation, name, buffers):
        self.owner, self.generation = owner, generation
        device = owner.pool.backing.device
        self.before = buffers.zeros((1,), torch.int64)
        m = generation.decoder.metadata
        metadata = SimpleNamespace(positions=m.indexed.positions, requests=m.indexed.requests,
            state_rows=m.recurrent.convolution_slots, starts=self.before)
        self.context = PreparedContext(owner.weights, generation.decoder.auxiliary, metadata, owner.pool,
            session=owner.session, name=name + ".context", buffers=buffers)

    def rebind_pages(self, pages):
        self.generation.rebind_pages(pages)

    def __call__(self):
        o, g = self.owner, self.generation
        slot = o.inputs.row_slot_ids[:1].long().clamp(0, o.requests.capacity - 1)
        self.before.copy_(o.requests.consumed_tokens[slot])
        g()
        self.context(g.decoder.metadata.recurrent.num_tokens)


class PreparedDFlashRound(PreparedRound):
    """Reuse GLM target metadata, exact KDA replay, filtering, rejection sampling and completion."""

    def __init__(self, owner, batch, width, name):
        self.owner, self.batch, self.width, self.name = owner, batch, width, name
        o, b, w = owner, batch, width
        device, blocks = o.pool.backing.device, o.inputs.page_tables.shape[1]
        from ignition.components.parallel_draft import prepare_sampling_scratch

        buffers = o.buffers.cursor()
        self.broadcast_buffers = o.sampling_buffers.cursor()
        self.sampling_scratch = prepare_sampling_scratch(b, w, o.requests.vocab_size, device, buffers=self.broadcast_buffers)
        self.rows, self.columns = torch.arange(b, device=device), torch.arange(w, device=device)
        self.offsets = torch.arange(b * w, device=device)
        self.state_columns = torch.arange(8 if w > 1 else 1, device=device)
        self.convolution_regions = convolution_regions(o.pool)
        m = DecoderMetadata.allocate(b * w, b, blocks, device)
        self.metadata = DecoderMetadata(m.token_ids, torch.arange(b * w, device=device),
            DeltaMetadata.allocate(b, device, verification_columns=8 if w > 1 else 1), m.indexed)
        self.validator = PreparedBatch(o.inputs, o.requests, DecoderMetadata.allocate(b, b, blocks, device),
            mode="decode", shard=o.weights.shard, pages=o.pool.memory_capacity[1:], shards=o.shards)
        self.validator.accept_unsampled = False
        self.validation = PreparedAllReduce(self.validator.validation_input, self.validator.validation_sum,
            runtime=o.collectives["runtime"], coordinator=o.collectives["coordinator"], session=o.session, name=name + ".validation")
        self.decoder = PreparedDecoder(o.weights, o.scratch.rows(b * w, "decode", sample_rows=b * w),
            o.pool.decoder, self.metadata, mode="decode", session=o.session, name=name + ".target", workspace=o.workspace,
            max_cache_pages=o.max_cache_pages, context=o.context, norm_kernels=o.norm_kernels,
            kda_records=o.kda_records if w > 1 else None, auxiliary_buffers=buffers, **o.collectives)
        first = self.decoder.layers[o.kda_records.layers[0]].attention
        self.kda_commit = (PreparedKDACommit(o.pool, o.kda_records, first.recurrent_plan,
            self.metadata.recurrent.state_indices[:, 0], self.metadata.recurrent.num_seqs,
            session=o.session, workspace=o.workspace) if w > 1 else None)
        self.before = buffers.zeros((b,), torch.int64)
        self.pending = buffers.zeros((b,), torch.int64)
        self.active = buffers.zeros((b,), torch.bool)
        self.candidates = buffers.zeros((b, w), torch.int64)
        self.drafts = buffers.zeros((b, w - 1), torch.int64)
        self.probabilities = buffers.zeros((b, w - 1, o.requests.vocab_size), torch.float32)
        self.valid = buffers.zeros((b, w - 1), torch.bool)
        self.seed, self.offset, self.base_offset, self.sampled = (buffers.zeros((b,), torch.int64) for _ in range(4))
        self.markov_token = buffers.zeros((b,), torch.int64)
        self.draft_metadata = Metadata.allocate(b * 8, b, blocks, device, samples=b * 8)
        self.draft_inputs = torch.full((b, 8), o.weights.dflash2.config.mask_token, dtype=torch.int64, device=device)
        self.draft = (PreparedDraft(o.weights, self.draft_metadata, o.pool, session=o.session, name=name + ".draft",
            workspace=o.workspace, buffers=buffers, layer_buffers=o.layer_buffers, **o.collectives) if w > 1 else None)
        self.draft_logits = buffers.zeros((b, 1, o.requests.vocab_size), torch.float32) if w > 1 else None
        metadata = SimpleNamespace(positions=self.metadata.indexed.positions, requests=self.metadata.indexed.requests,
            state_rows=self.metadata.recurrent.convolution_slots, starts=self.before)
        self.context_update = PreparedContext(o.weights, self.decoder.auxiliary, metadata, o.pool,
            session=o.session, name=name + ".context", buffers=buffers)
        self.operations = None

    def rebind_pages(self, pages):
        self.decoder.rebind_pages()
        self.validator.pages = self.owner.pool.memory_capacity[1:]

    def commit_state(self, slots, counts):
        if self.width > 1:
            super().commit_state(slots, counts)
        else:
            o = self.owner
            states = o.inputs.request_state_rows[slots.clamp(0, o.requests.capacity - 1), 0]
            save_hidden(self.decoder.scratch.hidden, o.pool.last_hidden, states, counts > 0)

    def __call__(self):
        o, b, w = self.owner, self.batch, self.width
        target, proposals, verify, commit = self.operations
        slots, safe, limits = self.begin_round()
        self.base_offset.copy_(torch.where(o.draft_generation[safe] == o.requests.generation[safe], o.draft_offset[safe], 0))
        self.seed.copy_(o.requests.seed.view(torch.int64)[safe] ^ DRAFT_RNG_DOMAIN)
        if self.draft is not None:
            self.draft_inputs[:, 0].copy_(self.pending)
            self.draft_metadata.set(self.before, self.active.int() * 8, o.inputs.request_state_rows[safe, 0],
                o.inputs.page_tables[safe], self.draft_inputs, o.inputs.live_rows)
            self.draft()
            self.markov_token.copy_(self.pending)
            for i, proposal in enumerate(proposals):
                self.draft.selector(self.draft_logits[:, 0], self.markov_token, i + 1)
                self.offset.copy_(self.base_offset + 4 * i)
                proposal()
                self.drafts[:, i].copy_(self.sampled)
                self.probabilities[:, i].copy_(proposal.probabilities[:, 0])
                self.valid[:, i].copy_(proposal.valid[:, 0] & self.active & (i < limits - 1))
                self.markov_token.copy_(self.sampled)
            slot_ids = torch.arange(o.requests.capacity, device=slots.device)
            matches = self.active[None] & (slots[None] == slot_ids[:, None])
            base = torch.where(matches, self.base_offset[None], 0).sum(1)
            o.draft_offset.copy_(torch.where(matches.any(1), base + 4 * (w - 1), o.draft_offset))
            o.draft_generation.copy_(torch.where(matches.any(1), o.requests.generation, o.draft_generation))
        self.candidates[:, 0].copy_(self.pending)
        self.candidates[:, 1:].copy_(self.drafts)
        self.target_metadata(slots, limits)
        self.decoder()
        target()
        verify()
        commit()
        self.commit_state(slots, commit.emitted_count)
        self.context_update(commit.emitted_count)
