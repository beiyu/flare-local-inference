"""Diagnostic-only deployed GLM intermediate snapshots."""

def install_trace(module, tensors, layer=0):
    import torch
    handles = []
    def save(name, value):
        if isinstance(value, torch.Tensor):
            tensors[name] = value.detach().cpu().contiguous().clone()
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                save(name + '.' + str(i), item)
    names = ['patch_embed', 'post_layernorm', 'downsample', 'merger.proj',
             'merger.post_projection_norm', 'merger.extra_activation_func',
             'merger.gate_up_proj', 'merger.act_fn', 'merger.down_proj']
    names += ['blocks.' + str(i) for i in range(24)]
    names += [f'blocks.{layer}.' + n for n in ['norm1', 'norm2', 'attn.qkv',
        'attn.apply_rotary_emb', 'attn.attn', 'attn.proj', 'mlp.gate_up_proj',
        'mlp.act_fn', 'mlp.down_proj']]
    modules = dict(module.named_modules())
    for name in names:
        handles.append(modules[name].register_forward_hook(
            lambda m, a, o, n=name: save(n, o)))
    handles.append(modules[f'blocks.{layer}.attn.attn'].register_forward_pre_hook(
        lambda m, a, kw: [save('attention.' + k, kw[k]) for k in ['query', 'key', 'value']] and None,
        with_kwargs=True))
    return handles


def check_trace(args):
    import json
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from b12x.preparation import PreparationSession
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.glm53.vision import PreparedVision, VisionInputs, VisionScratch, clamped_swiglu, spatial_rotary, spatial_downsample
    from ignition.recipes.glm53.vision_weights import VisionWeights
    from ignition.components.glm_vision_kernels import OriginalVisionKernels, original_rms_out, original_add_rms, original_qk
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    trace = load_file(str(args.reference), device='cuda:0')
    count = len(trace['pixels'])
    capacity = args.capacity
    weights = VisionWeights.load(inspect_indexed_checkpoint(args.checkpoint), device)
    kernels = OriginalVisionKernels(args.kernels, device)
    s, inputs, arena = VisionScratch.allocate(max(256, capacity), device), VisionInputs.allocate(device), ScratchArena(device)
    if capacity < 256:
        merged = {"merged", "merge_projected", "merge_gate_up", "merge_activated", "output"}
        s = VisionScratch(*(getattr(s,n)[:capacity//4 if n in merged else capacity] for n in s.__dataclass_fields__))
    def ref(name):
        value = trace[name]
        return value.reshape(count, -1) if value.numel() % count == 0 else value
    def compare(name, actual, key=None):
        target = trace[key or name]
        actual = actual[:count if target.numel() // count == actual[0].numel() else count // 4].reshape(-1).float()
        target = target.reshape(-1).float()
        delta = actual - target
        print(json.dumps(dict(stage=name, relative_l2=float(delta.norm()/target.norm().clamp_min(1e-12)),
            max_absolute=float(delta.abs().max()), unequal=int(torch.count_nonzero(delta)), total=target.numel())), flush=True)
    with torch.inference_mode(), PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        encoder = PreparedVision(weights, s, inputs, kernels=kernels, session=session, workspace=arena, name='glm-vision-trace')
        arena.finalize()
        encoder.positions[:count].copy_(trace['positions'])
        encoder.cu.copy_(torch.tensor([0,count], dtype=torch.int32, device=device))
        encoder.num_patches.fill_(count)
        encoder.num_features.fill_(count//4)
        s.patches[:count].copy_(trace['pixels'])
        encoder.patch(s.patches)
        compare('patch_embed', s.residual)
        # Each primitive receives the original deployed operator's input, so
        # upstream errors cannot conceal the first arithmetic disagreement.
        w, qkv, output, up, down = encoder.blocks[args.layer]
        s.residual[:count].copy_(ref('patch_embed' if args.layer == 0 else 'blocks.'+str(args.layer-1)))
        original_rms_out(s.residual, w.norm1, s.normalized, kernels.handle)
        compare(f'blocks.{args.layer}.norm1', s.normalized)
        s.normalized[:count].copy_(ref(f'blocks.{args.layer}.norm1'))
        qkv(s.normalized)
        compare(f'blocks.{args.layer}.attn.qkv.0', s.qkv)
        s.qkv[:count].copy_(ref(f'blocks.{args.layer}.attn.qkv.0'))
        q,k,v = s.qkv.view(capacity,3,16,64).unbind(1)
        s.query.copy_(q)
        s.key.copy_(k)
        original_qk(s.query,s.key,w.q_norm,w.k_norm,kernels.handle)
        s.query.copy_(spatial_rotary(s.query,encoder.positions,encoder.cos,encoder.sin))
        s.key.copy_(spatial_rotary(s.key,encoder.positions,encoder.cos,encoder.sin))
        s.value.copy_(v)
        compare('attention.query',s.query)
        compare('attention.key',s.key)
        compare('attention.value',s.value)
        s.query[:count].copy_(trace['attention.query'].reshape(count,16,64))
        s.key[:count].copy_(trace['attention.key'].reshape(count,16,64))
        s.value[:count].copy_(trace['attention.value'].reshape(count,16,64))
        encoder.attention()
        compare(f'blocks.{args.layer}.attn.attn',s.attended)
        s.attended[:count].copy_(ref(f'blocks.{args.layer}.attn.attn'))
        output(s.attended)
        compare(f'blocks.{args.layer}.attn.proj.0',s.projected)
        s.projected[:count].copy_(ref(f'blocks.{args.layer}.attn.proj.0'))
        s.normalized.copy_(s.residual)
        original_add_rms(s.normalized,s.projected,w.norm2,kernels.handle)
        compare(f'blocks.{args.layer}.norm2.0',s.normalized)
        compare(f'blocks.{args.layer}.norm2.1',s.projected)
        s.normalized[:count].copy_(ref(f'blocks.{args.layer}.norm2.0'))
        up(s.normalized)
        compare(f'blocks.{args.layer}.mlp.gate_up_proj.0',s.gate_up)
        s.gate_up[:count].copy_(ref(f'blocks.{args.layer}.mlp.gate_up_proj.0'))
        s.activated.copy_(clamped_swiglu(s.gate_up,10))
        compare(f'blocks.{args.layer}.mlp.act_fn',s.activated)
        s.activated[:count].copy_(ref(f'blocks.{args.layer}.mlp.act_fn'))
        down(s.activated)
        compare(f'blocks.{args.layer}.mlp.down_proj.0',s.projected)
        # The tail is also isolated from the transformer block accumulation.
        s.residual[:count].copy_(ref('blocks.23'))
        original_rms_out(s.residual,weights.norm,s.normalized,kernels.handle)
        compare('post_layernorm',s.normalized)
        s.normalized[:count].copy_(ref('post_layernorm'))
        s.merged.copy_(spatial_downsample(s.normalized,weights.downsample,weights.downsample_bias))
        compare('downsample',s.merged)
        # Reference convolution output is NCHW with 1x1 spatial dimensions.
        s.merged[:count//4].copy_(trace['downsample'].reshape(count//4,-1))
        encoder.merge_project(s.merged)
        compare('merger.proj.0',s.merge_projected)
        s.merge_projected[:count//4].copy_(trace['merger.proj.0'].reshape(count//4,-1))
        norm = F.layer_norm(s.merge_projected,(4096,),weights.merge_norm,weights.merge_norm_bias,1e-5)
        compare('merger.post_projection_norm',norm)
        activated = F.gelu(trace['merger.post_projection_norm'])
        compare('merger.extra_activation_func',activated)
        s.merge_projected[:count//4].copy_(trace['merger.extra_activation_func'])
        encoder.merge_up(s.merge_projected)
        compare('merger.gate_up_proj.0',s.merge_gate_up)
        s.merge_gate_up[:count//4].copy_(trace['merger.gate_up_proj.0'])
        s.merge_activated.copy_(clamped_swiglu(s.merge_gate_up,10))
        compare('merger.act_fn',s.merge_activated)
        s.merge_activated[:count//4].copy_(trace['merger.act_fn'])
        encoder.merge_down(s.merge_activated)
        compare('merger.down_proj.0',s.output)
        # Compare every transformer block in the complete eager chain too.
        encoder.patch(s.patches)
        for i,(w,qkv,output,up,down) in enumerate(encoder.blocks):
            original_rms_out(s.residual,w.norm1,s.normalized,kernels.handle)
            q,k,v=qkv(s.normalized).view(capacity,3,16,64).unbind(1)
            s.query.copy_(q)
            s.key.copy_(k)
            original_qk(s.query,s.key,w.q_norm,w.k_norm,kernels.handle)
            s.query.copy_(spatial_rotary(s.query,encoder.positions,encoder.cos,encoder.sin))
            s.key.copy_(spatial_rotary(s.key,encoder.positions,encoder.cos,encoder.sin))
            s.value.copy_(v)
            encoder.attention()
            attended=output(s.attended)
            s.normalized.copy_(s.residual)
            original_add_rms(s.normalized,attended,w.norm2,kernels.handle)
            s.residual.copy_(attended)
            s.activated.copy_(clamped_swiglu(up(s.normalized),10))
            s.residual.add_(down(s.activated))
            compare('blocks.'+str(i),s.residual)
        s.patches[:count].copy_(trace['pixels'])
        actual=encoder()
        compare('expected',actual)


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--capacity',type=int,default=256)
    parser.add_argument('--layer',type=int,default=0)
    parser.add_argument('--kernels',type=Path,required=True)
    check_trace(parser.parse_args())
