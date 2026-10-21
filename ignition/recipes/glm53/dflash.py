"""GLM DFlash2 parallel blocks, candidate paths, and accepted target feedback."""

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.grouped_convolution import PreparedGroupedConvolution
from ignition.components.linear import UnquantizedProjection
from ignition.components.mhc import PreparedMHCPost
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.window_attention import stage_windows


def norm(values, weight, epsilon):
    x = values.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * weight.float()).to(values.dtype)


class Rotary:
    def __init__(self, config, device):
        self.inverse = 1 / config.rope_theta ** (torch.arange(0, 128, 2, dtype=torch.float32, device=device) / 128)

    def __call__(self, values, positions):
        angles = positions.clamp_min(0).float()[:, None] * self.inverse[None]
        cosine, sine = angles.cos().bfloat16().float()[:, None], angles.sin().bfloat16().float()[:, None]
        left, right = values.float().chunk(2, -1)
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).to(values.dtype)


class PreparedFeedback:
    """Materialize deferred mHC post without changing the target's residual carry."""

    def __init__(self, config, layers, scratch, *, session, name, buffers):
        self.layers = config.target_layers
        self.auxiliary = buffers.zeros((scratch.embedding.shape[0], len(self.layers) * 4096), torch.bfloat16)
        self.streams = buffers.zeros(scratch.residual.shape, torch.bfloat16)
        self.posts = {}
        for index in self.layers:
            s = layers[index].scratch
            self.posts[index] = PreparedMHCPost(s.mlp_output, s.mlp_hc.residual,
                s.mlp_hc.post, s.mlp_hc.comb, self.streams, session=session, name=f"{name}.feedback{index}")

    def __call__(self, index, layer):
        if index in self.posts:
            s = layer.scratch
            self.posts[index](s.mlp_output, s.mlp_hc.residual, s.mlp_hc.post, s.mlp_hc.comb)
            column = self.layers.index(index) * 4096
            self.auxiliary[:, column:column + 4096].copy_(self.streams.float().mean(1).bfloat16())


class PreparedContext:
    """Write only accepted target features into the draft's independent circular KV."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, buffers):
        self.metadata, self.config = metadata, weights.dflash2.config
        self.auxiliary, self.weights = auxiliary, weights.dflash2
        c, d = self.config, weights.dflash2
        tokens, kh = auxiliary.shape[0], 8 // weights.shard.world_size
        self.projected = buffers.zeros((tokens, 4096), torch.bfloat16)
        self.normalized = buffers.zeros((tokens, 4096), torch.bfloat16)
        self.kv = buffers.zeros((tokens, 5 * 2 * kh * 128), torch.bfloat16)
        self.key, self.value = (buffers.zeros((tokens, kh, 128), torch.bfloat16) for _ in range(2))
        self.slots = buffers.zeros((tokens,), torch.int64).fill_(-1)
        self.project = UnquantizedProjection(auxiliary, d.projection, self.projected, session=session, name=name + ".project")
        self.normalize = PreparedRMSNorm(self.projected, d.hidden_norm, self.normalized,
            eps=c.epsilon, session=session, name=name + ".norm")
        self.project_kv = UnquantizedProjection(self.normalized, d.context_kv, self.kv, session=session, name=name + ".kv")
        self.rotary = Rotary(c, auxiliary.device)
        self.writers = tuple(PreparedPagedKVWriter(self.key, self.value, pool.region(f"dflash2.{i}.key"),
            pool.region(f"dflash2.{i}.value"), self.slots, session=session, name=f"{name}.{i}.write") for i in range(5))

    def __call__(self, counts):
        m, c = self.metadata, self.config
        offsets = m.positions - m.starts[m.requests]
        count = counts[m.requests]
        active = (m.positions >= 0) & (offsets < count) & (offsets >= (count - c.window).clamp_min(0))
        self.slots.copy_(torch.where(active, m.state_rows[m.requests] * c.window + m.positions.remainder(c.window), -1))
        self.normalize(self.project(self.auxiliary))
        self.project_kv(self.normalized)
        kv = self.kv.view(self.kv.shape[0], 5, 2, *self.key.shape[1:])
        for i, writer in enumerate(self.writers):
            self.key.copy_(self.rotary(norm(kv[:, i, 0], self.weights.layers[i]["k_norm"], c.epsilon), m.positions))
            self.value.copy_(kv[:, i, 1])
            writer()

class DraftAttention:
    def __init__(self, weights, source, metadata, pool, config, shard, index, *, session, name, workspace, buffers):
        self.weights, self.metadata, self.config = weights, metadata, config
        tokens, qh, kh = source.shape[0], 32 // shard.world_size, 8 // shard.world_size
        self.qkv = buffers.zeros((tokens, (qh + 2 * kh) * 128), torch.bfloat16)
        self.query = buffers.zeros((tokens, qh, 128), torch.bfloat16)
        self.key, self.value = (buffers.zeros((tokens, kh, 128), torch.bfloat16) for _ in range(2))
        self.attended = buffers.zeros(self.query.shape, torch.bfloat16)
        self.project = UnquantizedProjection(source, weights["qkv"], self.qkv, session=session, name=name + ".qkv")
        self.output = buffers.zeros(source.shape, torch.bfloat16)
        self.project_output = UnquantizedProjection(self.attended.flatten(1), weights["output"], self.output,
            session=session, name=name + ".output")
        self.rotary = Rotary(config, source.device)
        self.history = tuple(pool.region(f"dflash2.{index}.{kind}") for kind in ("key", "value"))
        rows = metadata.lengths.numel()
        pages = (config.window + config.block_size + 127) // 128
        self.staged = tuple(buffers.zeros((rows * pages, 128, kh, 128), torch.bfloat16) for _ in range(2))
        self.pages = torch.arange(rows * pages, dtype=torch.int32, device=source.device).view(rows, pages)
        self.lengths = buffers.zeros((rows,), torch.int32)
        self.attention = PreparedPagedAttention(self.query, *self.staged, self.pages, self.lengths,
            metadata.query_start_loc, self.attended, mode="verify", width=config.block_size,
            window=config.window - 1, causal=False, session=session, workspace=workspace, name=name + ".attention")

    def __call__(self, source):
        m, c = self.metadata, self.config
        q, k, v = self.project(source).split((self.query.shape[1] * 128, self.key.shape[1] * 128, self.key.shape[1] * 128), -1)
        self.query.copy_(self.rotary(norm(q.view_as(self.query), self.weights["q_norm"], c.epsilon), m.positions))
        self.key.copy_(self.rotary(norm(k.view_as(self.key), self.weights["k_norm"], c.epsilon), m.positions))
        self.value.copy_(v.view_as(self.value))
        stage_windows(self.history[0], self.key, self.staged[0], m)
        stage_windows(self.history[1], self.value, self.staged[1], m)
        self.lengths.copy_((m.starts.clamp(max=c.window) + m.lengths).int())
        self.attention()
        return self.project_output(self.attended.flatten(1))


class CandidateSelector:
    """Realized conditional logits on the original top-16 candidate paths."""

    def __init__(self, weights, hidden, logits, *, session, name, buffers):
        self.weights, self.logits = weights.selector, logits
        self.hidden = buffers.zeros((*hidden.shape[:-1], 256), torch.bfloat16)
        self.project = UnquantizedProjection(hidden.flatten(0, 1), self.weights["hidden"], self.hidden.flatten(0, 1),
            session=session, name=name + ".hidden")
        self.candidates = buffers.zeros((*hidden.shape[:-1], 16), torch.int64)
        self.unary = buffers.zeros(self.candidates.shape, logits.dtype)

    def prepare(self, hidden, logits):
        self.project(hidden.flatten(0, 1))
        unary, candidates = logits.topk(16, dim=-1)
        self.candidates.copy_(candidates)
        self.unary.copy_(unary)

    def __call__(self, output, previous, position):
        candidate = self.candidates[:, position]
        # Original BF16 products/reduction precede the float32 sampling workspace.
        predecessor = self.weights["predecessor"][previous]
        successor = self.weights["successor"][candidate]
        hidden = (predecessor * self.hidden[:, position]).bfloat16()
        scores = torch.bmm(hidden[:, None], successor.transpose(1, 2)).squeeze(1).bfloat16()
        scores = (self.unary[:, position] + scores).bfloat16()
        output.fill_(-float("inf"))
        output.scatter_(1, candidate, scores.float())


class PreparedDraft:
    def __init__(self, weights, metadata, pool, *, session, name, workspace, runtime, vocabulary_runtime,
                 coordinator, buffers, layer_buffers):
        c, d = weights.dflash2.config, weights.dflash2
        self.metadata, self.config = metadata, c
        tokens = metadata.token_ids.numel()
        shape = (tokens, 4096)
        self.embedding, self.residual, self.normalized = (buffers.zeros(shape, torch.bfloat16) for _ in range(3))
        self.hidden = buffers.zeros(shape, torch.bfloat16)
        self.local_logits = buffers.zeros((tokens, 154880 // weights.shard.world_size), torch.bfloat16)
        self.logits = buffers.zeros((tokens, 154880), torch.bfloat16)
        self.embed = PreparedEmbedding(weights.tree.model.language_model.embed_tokens.weight, metadata.token_ids,
            self.embedding, num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.layers = []
        for i, w in enumerate(d.layers):
            b, prefix = layer_buffers.cursor(), f"{name}.layer{i}"
            before = PreparedRMSNorm(self.residual, w["input_norm"], self.normalized, eps=c.epsilon,
                session=session, name=prefix + ".norm")
            convs = tuple(PreparedGroupedConvolution(self.normalized, w[kind], block_size=c.block_size,
                group_size=c.group_size, session=session, name=prefix + "." + kind,
                buffers={key: b.zeros((tokens, width), torch.bfloat16) for key, width in
                    (("coefficients", 1024), ("before", 4096), ("after", 4096))}) for kind in ("attention_conv", "mlp_conv"))
            attention = DraftAttention(w, self.normalized, metadata, pool, c, weights.shard, i,
                session=session, name=prefix + ".attention", workspace=workspace, buffers=b)
            reduced = b.zeros(shape, torch.bfloat16)
            reduce_attention = PreparedAllReduce(attention.output, reduced, runtime=runtime,
                coordinator=coordinator, session=session, name=prefix + ".attention_reduce")
            post = PreparedRMSNorm(self.residual, w["post_norm"], self.normalized, eps=c.epsilon,
                session=session, name=prefix + ".post_norm")
            mlp_scratch = MLPScratch.allocate(tokens, 4096, 12288 // weights.shard.world_size, self.embedding.device, buffers=b)
            mlp = PreparedMLP(self.normalized, w["mlp"], mlp_scratch.gate_up, mlp_scratch.intermediate, mlp_scratch.output,
                session=session, name=prefix + ".mlp", workspace=workspace, round_silu=False)
            reduce_mlp = PreparedAllReduce(mlp_scratch.output, reduced, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".mlp_reduce")
            self.layers.append((before, convs, attention, reduce_attention, post, mlp, reduce_mlp, reduced))
        self.norm = PreparedRMSNorm(self.residual, d.norm, self.hidden, eps=c.epsilon, session=session, name=name + ".norm")
        self.head = PreparedVocabularyProjection(self.hidden, weights.tree.lm_head.weight, self.local_logits, self.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")
        rows = metadata.lengths.numel()
        self.selector = CandidateSelector(d, self.hidden.view(rows, c.block_size, 4096), self.logits,
            session=session, name=name + ".selector", buffers=buffers)

    def __call__(self):
        self.residual.copy_(self.embed())
        for before, convs, attention, attention_reduce, post, mlp, mlp_reduce, reduced in self.layers:
            x = before(self.residual)
            attention_reduce(attention(convs[0].prepare(x)))
            self.residual.add_(convs[0].finish(reduced))
            x = post(self.residual)
            mlp_reduce(mlp(convs[1].prepare(x)))
            self.residual.add_(convs[1].finish(reduced))
        self.norm(self.residual)
        self.head(self.hidden)
        rows = self.metadata.lengths.numel()
        self.selector.prepare(self.hidden.view(rows, self.config.block_size, 4096),
            self.logits.view(rows, self.config.block_size, -1))
