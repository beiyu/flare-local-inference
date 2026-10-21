"""Record CPU image tensors from the deployment's actual GLM processor.

This qualifies frontend pixels and patch order only. It does not qualify the
vision tower, decoder injection, GPU graphs, or GLM image serving.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def pattern(width, height, seed):
    state = seed
    data = bytearray(width * height * 3)
    for i in range(len(data)):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        data[i] = state & 255
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def digest(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().contiguous().numpy()
    return hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest()


def generate(processor_path, config_path, output):
    spec = importlib.util.spec_from_file_location("glm_deployment_processor", processor_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(config_path.read_text())["image_processor"]
    torch.set_num_threads(1)
    cases = []
    specs = [
        ("identity", [(112, 112, 1)], 1),
        ("tiny_upscale", [(1, 1, 2)], 1),
        ("upscale_pad", [(29, 30, 3)], 1),
        ("pad_without_upscale", [(117, 130, 4)], 1),
        ("asymmetric", [(273, 289, 5)], 1),
        ("budget_fit", [(4000, 3000, 6)], 1),
        ("narrow", [(5000, 1, 7)], 1),
        ("expanded_alignment", [(256, 80, 8)], 2),
        ("mixed", [(112, 112, 9), (29, 30, 10)], 1),
    ]
    for name, inputs, expand in specs:
        options = {**config, "max_image_tokens": 2048, "patch_expand_factor": expand}
        processor = module.Glm5NextImageProcessor(**options)
        rgbs = [pattern(*item) for item in inputs]
        result = processor(images=[Image.fromarray(rgb) for rgb in rgbs], return_tensors="pt")
        grids = result["image_grid_thw"].cpu()
        canvas_hashes, positions, cu = [], [], [0]
        for rgb, grid in zip(rgbs, grids):
            _, h, w = map(int, grid)
            height, width = rgb.shape[:2]
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
            canvas = module._resize_or_pad(tensor, h * 14, w * 14, "pad", processor.resample,
                processor.resize, allow_upscale=(2 * height * width < 16 * 2 * 28**2))
            canvas_hashes.append(digest(canvas[0].permute(1, 2, 0)))
            rows = torch.arange(h).unsqueeze(1).expand(-1, w)
            cols = torch.arange(w).unsqueeze(0).expand(h, -1)
            positions.append(torch.stack([v.reshape(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3).flatten()
                for v in (rows, cols)], dim=-1).to(torch.int64))
            cu.append(cu[-1] + h * w)
        pixels = result["pixel_values"].contiguous()
        cases.append(dict(name=name,
            settings=dict(min_image_tokens=16, max_image_tokens=2048, patch_expand_factor=expand),
            inputs=[dict(width=w, height=h, seed=seed) for w, h, seed in inputs],
            grid_thw=grids.tolist(), pixel_values_shape=list(pixels.shape),
            resized_rgb_sha256=canvas_hashes, pixel_values_sha256=digest(pixels),
            position_ids_sha256=digest(torch.cat(positions)), cu_seqlens=cu))
        print(f"{name}: {list(pixels.shape)}", flush=True)
    output.write_text(json.dumps(dict(reference={
        "processor_sha256": hashlib.sha256(processor_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "Pillow")},
    }, cases=cases), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.processor_path, args.config_path, args.output)
