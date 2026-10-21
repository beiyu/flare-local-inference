"""Build an independent GLM migration checkpoint with deployed BF16 donors and template.

Runs on CPU. Source weights and donor files are read-only. Unchanged shards are
symlinked; shards containing transplanted tensors are copied before range writes.
Keep the source checkpoint mounted at the same path through preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from ignition.preparation.checkpoint import inspect_indexed_checkpoint

DONOR_REVISION = "745aac2ff0f10acf961f396df3f9418598aa7327"
LAYERS = tuple(range(15, 46))


def plan(checkpoint: Path, donors: Path, template: Path):
    checkpoint, donors, template = (p.expanduser().resolve() for p in (checkpoint, donors, template))
    if not template.is_file() or not template.read_text().strip():
        raise ValueError("the deployed custom chat template must be a nonempty UTF-8 file")
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("architectures") != ["Glm5NextForConditionalGeneration"] or config.get("quantization_config", {}).get("quant_method") != "exl3":
        raise ValueError("expected the deployed GLM-5.3 EXL3 checkpoint")
    manifest = inspect_indexed_checkpoint(checkpoint)
    description = json.loads((donors / "MANIFEST.json").read_text())
    if description.get("donor_sha") != DONOR_REVISION:
        raise ValueError("donor revision differs from the deployed transplant")
    entries = description.get("layers", {})
    if set(entries) != {str(layer) for layer in LAYERS}:
        raise ValueError("donor manifest must contain every layer 15 through 45, including MTP")
    patches = []
    for layer in LAYERS:
        name = f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
        tensor = manifest.tensors.get(name)
        donor = donors / f"L{layer}.bin"
        entry = entries[str(layer)]
        if tensor is None or tensor.dtype != "BF16" or (entry.get("key"), entry.get("dtype"), tuple(entry.get("shape", [])), entry.get("nbytes")) != (name, tensor.dtype, tensor.shape, tensor.nbytes):
            raise ValueError(f"L{layer}: donor and checkpoint BF16 tensor geometry differ")
        if donor.stat().st_size != tensor.nbytes:
            raise ValueError(f"L{layer}: donor byte count differs")
        with donor.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"L{layer}: donor SHA256 differs from its deployed manifest")
        patches.append((tensor, donor, digest))
    rewritten = {tensor.shard for tensor, _, _ in patches}
    copied_bytes = sum(shard.file_size for shard in manifest.shards if shard.path in rewritten)
    return manifest, patches, template, copied_bytes


def create(checkpoint: Path, donors: Path, template: Path, output: Path):
    manifest, patches, template, copied_bytes = plan(checkpoint, donors, template)
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < copied_bytes + template.stat().st_size:
        raise ValueError(f"output filesystem needs at least {copied_bytes} free bytes for copied shards")
    output.mkdir()
    rewritten = {tensor.shard for tensor, _, _ in patches}
    # Metadata is independent, including configuration and exact tokenizer assets.
    for source in manifest.snapshot.iterdir():
        if source.is_file() and not source.name.endswith(".safetensors"):
            shutil.copyfile(source, output / source.name)
    for shard in manifest.shards:
        source, destination = manifest.snapshot / shard.path, output / shard.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if shard.path in rewritten:
            shutil.copyfile(source, destination)
        else:
            destination.symlink_to(source.resolve())
    for tensor, donor, digest in patches:
        # Recheck the donor while copying; never write through a source symlink.
        destination = output / tensor.shard
        if destination.is_symlink():
            raise ValueError("refusing to write a donor through a symlink")
        hashed = hashlib.sha256()
        with destination.open("r+b") as target, donor.open("rb") as source:
            target.seek(tensor.offset)
            while block := source.read(8 << 20):
                target.write(block); hashed.update(block)
        if hashed.hexdigest() != digest:
            raise ValueError(f"{donor}: changed during transplant")
    shutil.copyfile(template, output / "chat_template.jinja")
    # Transformers may prefer an embedded template over chat_template.jinja.
    tokenizer_config = output / "tokenizer_config.json"
    if tokenizer_config.is_file():
        config = json.loads(tokenizer_config.read_text())
        config["chat_template"] = template.read_text()
        tokenizer_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    exported = inspect_indexed_checkpoint(output)
    if dict(exported.tensors) != dict(manifest.tensors):
        raise ValueError("transplant changed tensor names, geometry, or byte extents")
    for tensor, _, digest in patches:
        hashed = hashlib.sha256()
        with (output / tensor.shard).open("rb") as source:
            source.seek(tensor.offset)
            remaining = tensor.nbytes
            while remaining:
                block = source.read(min(remaining, 8 << 20))
                if not block: raise ValueError("truncated transplanted tensor")
                hashed.update(block); remaining -= len(block)
        if hashed.hexdigest() != digest:
            raise ValueError(f"{tensor.name}: copied tensor does not match deployed donor")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "create"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--donors", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "create":
        if args.output is None: parser.error("create requires --output")
        print(create(args.checkpoint, args.donors, args.template, args.output))
    else:
        manifest, patches, _, copied_bytes = plan(args.checkpoint, args.donors, args.template)
        print(json.dumps({"checkpoint": str(manifest.snapshot), "donor_layers": len(patches),
                          "copied_shard_bytes": copied_bytes, "custom_template": str(args.template)}, indent=2))


if __name__ == "__main__":
    main()
