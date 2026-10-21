"""Validate scheduled GLM images and publish generation-scoped feature windows."""

import torch

from ignition.components.vision_delivery import PreparedFeaturePublication


class PreparedVisionGeneration:
    def __init__(self, encoder, state, completion, frame, sequence, *, buffers=None):
        self.encoder, self.state = encoder, state
        device = completion.records.device
        self.errors = (torch.zeros(1, dtype=torch.int32, device=device) if buffers is None
                       else buffers.zeros((1,), torch.int32))
        self.publish = PreparedFeaturePublication(state, encoder.inputs, completion, frame, sequence)
        n = encoder.scratch.patches.shape[0]
        self.patch_rows = torch.arange(n, device=device)
        self.feature_rows = torch.arange(n // 4, device=device)

    def __call__(self):
        e, s = self.encoder, self.state
        i, r = e.inputs, s.requests
        n = e.scratch.patches.shape[0]
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, features, offset = i.vision_num_patches, i.vision_num_features, i.vision_feature_offset
        valid = (i.vision_slot >= 0) & (i.vision_slot < r.capacity) & (i.vision_generation > 0)
        valid &= (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid &= (count > 0) & (count <= n) & (features > 0) & (features <= n // 4) & (count == features * 4)
        valid &= (offset >= 0) & (offset <= s.arena.shape[1]) & (offset + features <= s.arena.shape[1])
        valid &= (offset == 0) | ((s.generation[slot] == i.vision_generation) & (s.count[slot] == offset))
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count)
        live = self.patch_rows < count
        positions = i.vision_position_ids[:n]
        valid &= (~live[:, None] | ((positions >= 0) & (positions < n))).all()
        valid &= (~live[:, None] | torch.isfinite(i.vision_pixel_values[:n])).all()
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        self.errors.copy_(~valid)
        e.scratch.patches.copy_(torch.where(valid & live[:, None], i.vision_pixel_values[:n], 0))
        e.positions.copy_(torch.where(valid & live[:, None], positions, 0))
        e.cu.copy_(torch.where(valid, i.vision_cu_seqlens, 0))
        e.num_patches.copy_(torch.where(valid, count, 0))
        e.num_features.copy_(torch.where(valid, features, 0))
        output = e()
        feature_live = self.feature_rows < e.num_features
        self.errors.copy_(self.errors | (~((~feature_live[:, None]) | torch.isfinite(output)).all()).int() * 64)
        self.publish(output, e.num_features, self.errors)
