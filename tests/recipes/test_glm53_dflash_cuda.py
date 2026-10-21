"""Real DFlash2 weights: accepted circular context and compiled native backbone replay.

These component checks do not establish complete TP2 target-state or HTTP parity.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("b12x")
pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device and real GLM/DFlash2 checkpoints")


def norm(x, weight, epsilon):
    values = x.float()
    return (values * (values.square().mean(-1, keepdim=True) + epsilon).rsqrt() * weight.float()).bfloat16()


class Rotary:
    def __init__(self, config, device):
        self.frequency = 1.0 / (config.rope_theta ** (torch.arange(0, 128, 2, device=device).float() / 128))

    def __call__(self, values, positions):
        angles = positions.clamp_min(0).float()[:, None] * self.frequency
        cos, sin = angles.cos().bfloat16().float()[:, None], angles.sin().bfloat16().float()[:, None]
        a, b = values.float().chunk(2, -1)
        return torch.cat((a * cos - b * sin, a * sin + b * cos), -1).bfloat16()


@torch.no_grad()
def test_context_acceptance_window_wrap_and_inactive_rows():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    from types import SimpleNamespace
    import torch.nn.functional as F
    from b12x.preparation import PreparationSession
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.paged_batch import Metadata
    from ignition.components.state import StateLayout, StateStorage
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.recipes.glm53.dflash_weights import DraftWeights
    from ignition.recipes.glm53.dflash import PreparedContext, DraftAttention

    torch.set_grad_enabled(False)
    torch.manual_seed(71237)
    device='cuda:0'
    shard=TensorParallelShard(0,2)
    path=Path(os.environ["FLARE_GLM_DFLASH2_CHECKPOINT"])
    w=DraftWeights.load(path, device, shard)
    weights=SimpleNamespace(dflash2=w,shard=shard)
    layout=StateLayout.from_specs([(name,shape,torch.bfloat16) for name,shape,_ in w.config.regions(2)])
    pool=StateStorage.allocate(layout,4,device)
    for i in range(5):
     for kind in ('key','value'):
      t=pool.region(f'dflash2.{i}.{kind}')
      t.copy_(torch.randn(t.shape,dtype=t.dtype,device=device)*.1)
    initial=pool.backing.clone()
    expected_pool=StateStorage(backing=initial.clone(),layout=layout,memory_capacity=pool.memory_capacity)
    b,tokens=2,16
    m=Metadata.allocate(tokens,b,16,device,samples=tokens)
    starts=torch.tensor([2047,4095],device=device)
    lengths=torch.tensor([8,8],device=device,dtype=torch.int32)
    rows=torch.tensor([1,2],device=device)
    pages=torch.zeros((b,16),dtype=torch.int32,device=device)
    ids=torch.zeros((b,8),dtype=torch.int64,device=device)
    live=torch.tensor([2],dtype=torch.int32,device=device)
    m.set(starts,lengths,rows,pages,ids,live)
    aux=torch.randn((tokens,5*4096),dtype=torch.bfloat16,device=device)*.1
    source=torch.randn((tokens,4096),dtype=torch.bfloat16,device=device)*.1
    counts=torch.tensor([3,0],dtype=torch.int32,device=device)
    buffers=GraphBufferPool(device)
    with PreparationSession(device=device,autotune=False) as session:
     arena=ScratchArena(device)
     cursor=buffers.cursor()
     context=PreparedContext(weights,aux,m,pool,session=session,name='context',buffers=cursor)
     attention=DraftAttention(w.layers[0],source,m,pool,w.config,shard,0,session=session,
      name='attention',workspace=arena,buffers=cursor)
     arena.finalize()
     context(counts)
     torch.cuda.synchronize()
     projected=norm(F.linear(aux,w.projection),w.hidden_norm,w.config.epsilon)
     expected_kv=F.linear(projected,w.context_kv).view(tokens,5,2,4,128)
     torch.testing.assert_close(context.normalized, projected, rtol=0, atol=0)
     print('context_relative_l2',float((context.normalized.float()-projected.float()).norm()/projected.float().norm()),flush=True)
     for i in range(5):
      expected_key=Rotary(w.config,aux.device)(norm(expected_kv[:,i,0],w.layers[i]['k_norm'],w.config.epsilon),m.positions)
      for step in range(int(counts[0])):
       torch.testing.assert_close(pool.region(f'dflash2.{i}.key')[1, (starts[0]+step)%2048], expected_key[step], rtol=0, atol=0)
       torch.testing.assert_close(pool.region(f'dflash2.{i}.value')[1, (starts[0]+step)%2048], expected_kv[step,i,1], rtol=0, atol=0)
       expected_pool.region(f'dflash2.{i}.key')[1,(starts[0]+step)%2048].copy_(expected_key[step])
       expected_pool.region(f'dflash2.{i}.value')[1,(starts[0]+step)%2048].copy_(expected_kv[step,i,1])
      a=pool.region(f'dflash2.{i}.key')[1, starts[0].item()%2048]
      print('context_key',i,float((a.float()-expected_key[0].float()).abs().max()),flush=True)
     assert torch.equal(pool.backing[2],initial[2]),'inactive context changed'
     assert torch.equal(pool.backing,expected_pool.backing),'accepted or rejected context bytes differ'
     graph=torch.cuda.CUDAGraph()
     with session.capture(),torch.cuda.graph(graph): context(counts)
     pool.backing.copy_(initial)
     graph.replay()
     torch.cuda.synchronize()
     assert torch.equal(pool.backing[2],initial[2]),'inactive captured context changed'
     assert torch.equal(pool.backing,expected_pool.backing),'captured context bytes differ'
     starts=starts+counts
     lengths=torch.tensor([8,0],dtype=torch.int32,device=device)
     m.set(starts,lengths,rows,pages,ids,live)
     attention(source)
     torch.cuda.synchronize()
     q,k,v=F.linear(source,w.layers[0]['qkv']).split((2048,512,512),-1)
     q=Rotary(w.config,source.device)(norm(q.view(tokens,16,128),w.layers[0]['q_norm'],w.config.epsilon),m.positions)
     k=Rotary(w.config,source.device)(norm(k.view(tokens,4,128),w.layers[0]['k_norm'],w.config.epsilon),m.positions)
     v=v.view(tokens,4,128)
     positions=torch.arange(starts[0]-2048,starts[0]+8,device=device)
     hk=pool.region('dflash2.0.key')[1,positions[:2048]%2048]
     hv=pool.region('dflash2.0.value')[1,positions[:2048]%2048]
     kk=torch.cat((hk,k[:8])).transpose(0,1).repeat_interleave(4,0)
     vv=torch.cat((hv,v[:8])).transpose(0,1).repeat_interleave(4,0)
     mask=positions[None,:]>=m.positions[:8,None]-2047
     y=F.scaled_dot_product_attention(q[:8].transpose(0,1),kk,vv,attn_mask=mask)
     expected=F.linear(y.transpose(0,1).reshape(8,-1),w.layers[0]['output'])
     error=(attention.output[:8].float()-expected.float())
     print('attention_relative_l2',float(error.norm()/expected.float().norm()),'max_abs',float(error.abs().max()),flush=True)
     first=attention.output.clone()
     attention(source)
     torch.cuda.synchronize()
     torch.testing.assert_close(first,attention.output,rtol=0,atol=0)
     print('attention_repeat_max_abs',float((first-attention.output).float().abs().max()),flush=True)
     assert float(error.norm()/expected.float().norm()) < .002
     print('probe_complete',flush=True)

@torch.no_grad()
def test_compiled_backbone_native_replay_preserves_logits_and_allocations():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    from types import SimpleNamespace as NS
    import torch.nn.functional as F
    from b12x.preparation import PreparationSession
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.weights import DirectWeightLoader
    from ignition.components.paged_batch import Metadata
    from ignition.components.state import StateLayout, StateStorage
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import GraphBufferPool, ScratchArena
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.graphs import PreparedGraphBundle
    from ignition.recipes.glm53.dflash_weights import DraftWeights
    from ignition.recipes.glm53.dflash import PreparedDraft

    torch.set_grad_enabled(False)
    torch.manual_seed(71238)
    device='cuda:0'
    shard=TensorParallelShard(0,1)
    checkpoint=Path(os.environ["FLARE_EXL3_CHECKPOINT"])
    path=Path(os.environ["FLARE_GLM_DFLASH2_CHECKPOINT"])
    w=DraftWeights.load(path,device,shard)
    with DirectWeightLoader(inspect_indexed_checkpoint(checkpoint),device,prefixes=('model.language_model.embed_tokens.','lm_head.')) as loader:
     embedding,head=loader.load_tensors(('model.language_model.embed_tokens.weight','lm_head.weight'))
    weights=NS(dflash2=w,shard=shard,tree=NS(model=NS(language_model=NS(embed_tokens=NS(weight=embedding))),lm_head=NS(weight=head)))
    pool=StateStorage.allocate(StateLayout.from_specs([(n,s,torch.bfloat16) for n,s,_ in w.config.regions(1)]),3,device)
    for i in range(5):
     for kind in ('key','value'):
      t=pool.region(f'dflash2.{i}.{kind}')
      t.copy_(torch.randn(t.shape,dtype=t.dtype,device=device)*.1)
    m=Metadata.allocate(8,1,16,device,samples=8)
    starts=torch.tensor([2050],device=device)
    lengths=torch.tensor([8],device=device,dtype=torch.int32)
    rows=torch.tensor([1],device=device)
    pages=torch.zeros((1,16),dtype=torch.int32,device=device)
    ids=torch.full((1,8),154856,dtype=torch.int64,device=device)
    ids[0,0]=23456
    m.set(starts,lengths,rows,pages,ids,torch.tensor([1],dtype=torch.int32,device=device))
    buffers,layers=GraphBufferPool(device),GraphBufferPool(device)
    with PreparationSession(device=device,autotune=False) as session:
     arena=ScratchArena(device)
     draft=PreparedDraft(weights,m,pool,session=session,name='draft',workspace=arena,runtime=None,vocabulary_runtime=None,
      coordinator=None,buffers=buffers.cursor(),layer_buffers=layers)
     arena.finalize()
     draft()
     torch.cuda.synchronize()
     print('eager draft built',flush=True)
     def conv(x,coeff,base,side):
      x=x.view(-1,8,4096)
      d=coeff.view(-1,8,2,2,256)[:,:,side].repeat_interleave(16,-1)
      out=torch.zeros_like(x)
      for tap in range(2): out[:,tap:]+=(base[side,tap]+d[:,tap:,tap])*x[:,:8-tap]
      return out.reshape(-1,4096)
     residual=F.embedding(m.token_ids,embedding)
     positions=torch.arange(starts[0]-2048,starts[0]+8,device=device)
     mask=positions[None,:]>=m.positions[:,None]-2047
     for i,ww in enumerate(w.layers):
      x=norm(residual,ww['input_norm'],1e-5)
      coeff=F.linear(x,ww['attention_conv']['projection'])
      x=conv(x,coeff,ww['attention_conv']['base'],0)
      q,k,v=F.linear(x,ww['qkv']).split((4096,1024,1024),-1)
      q=Rotary(w.config,x.device)(norm(q.view(8,32,128),ww['q_norm'],1e-5),m.positions)
      k=Rotary(w.config,x.device)(norm(k.view(8,8,128),ww['k_norm'],1e-5),m.positions)
      v=v.view(8,8,128)
      hk=pool.region(f'dflash2.{i}.key')[1,positions[:2048]%2048]
      hv=pool.region(f'dflash2.{i}.value')[1,positions[:2048]%2048]
      kk=torch.cat((hk,k)).transpose(0,1).repeat_interleave(4,0)
      vv=torch.cat((hv,v)).transpose(0,1).repeat_interleave(4,0)
      y=F.scaled_dot_product_attention(q.transpose(0,1),kk,vv,attn_mask=mask)
      attn=F.linear(y.transpose(0,1).reshape(8,-1),ww['output'])
      residual=(residual+conv(attn,coeff,ww['attention_conv']['base'],1)).bfloat16()
      x=norm(residual,ww['post_norm'],1e-5)
      coeff=F.linear(x,ww['mlp_conv']['projection'])
      x=conv(x,coeff,ww['mlp_conv']['base'],0)
      gate,up=F.linear(x,ww['mlp'].gate_up).chunk(2,-1)
      mlp=F.linear((F.silu(gate.float())*up.float()).bfloat16(),ww['mlp'].down)
      residual=(residual+conv(mlp,coeff,ww['mlp_conv']['base'],1)).bfloat16()
     expected=norm(residual,w.norm,1e-5)
     print('backbone_relative_l2',float((draft.hidden.float()-expected.float()).norm()/expected.float().norm()),
      'max_abs',float((draft.hidden.float()-expected.float()).abs().max()),flush=True)
     expected_logits=F.linear(expected,head)
     print('head_relative_l2',float((draft.logits.float()-expected_logits.float()).norm()/expected_logits.float().norm()),
      'argmax_matches',int((draft.logits.argmax(-1)==expected_logits.argmax(-1)).sum()),flush=True)
     assert torch.equal(draft.logits.argmax(-1),expected_logits.argmax(-1))
     initial_hidden,initial_logits=draft.hidden.clone(),draft.logits.clone()
     compiled=compile_forward(draft)
     compiled()
     torch.cuda.synchronize()
     torch.testing.assert_close(draft.hidden,initial_hidden,rtol=0,atol=0)
     torch.testing.assert_close(draft.logits,initial_logits,rtol=0,atol=0)
     print('compiled_hidden_max_abs',float((draft.hidden-initial_hidden).float().abs().max()),
      'compiled_logits_max_abs',float((draft.logits-initial_logits).float().abs().max()),flush=True)
     graph=torch.cuda.CUDAGraph()
     with session.capture(),torch.cuda.graph(graph): compiled()
     bundle=PreparedGraphBundle(device=0,graphs={'draft':graph},buffers=dict(logits=draft.logits),resources=(draft,compiled,buffers,layers))
     worker=bundle.start_worker()
     keys=('allocation.all.allocated','allocation.all.freed','segment.all.allocated','segment.all.freed')
     before=tuple(torch.cuda.memory_stats()[key] for key in keys)
     ticket=worker.submit(0,repetitions=3)
     done=worker.poll(timeout_ms=120000)
     assert done is not None and done.ticket==ticket and done.error is None
     after=tuple(torch.cuda.memory_stats()[key] for key in keys)
     assert before==after
     torch.testing.assert_close(draft.hidden,initial_hidden,rtol=0,atol=0)
     torch.testing.assert_close(draft.logits,initial_logits,rtol=0,atol=0)
     print('native_hidden_max_abs',float((draft.hidden-initial_hidden).float().abs().max()),
      'native_logits_max_abs',float((draft.logits-initial_logits).float().abs().max()),'allocation_stable',before==after,flush=True)
     worker.close()
     graph.reset()
     print('backbone_probe_complete',flush=True)
