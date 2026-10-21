"""Validate the unchanged deployed CUDA images used during GLM preparation."""

import hashlib
import json
from pathlib import Path

MANIFEST = {
    "version": 1,
    "architecture": "sm121",
    "source_sha256": "cd14cceedf7799afff915fcfe52afb051b5d6c7e9040bd936bc8ad23bb5d4ae0",
    "images": {
        "rms.cubin": "cd24313368f43449aacece910684bed7f75e604c34fca1f50c57a20653d77a01",
        "qk.cubin": "5896acae8bb199bbea12e0cdc87c4015697cea9162a4ab9d022a0b9e9ccd5738",
    },
}
SYMBOLS = {
    "rms": "_ZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfii",
    "add": "_ZN4vllm25fused_add_rms_norm_kernelIN3c108BFloat16ELi8ELb1EEENSt9enable_ifIXaagtT0_Li0Esr4vllm12_typeConvertIT_EE6existsEvE4typeEPS4_lS7_PKS4_fiil",
    "qk": "_fused_q_kv_rmsnorm_kernel",
}
# Captured from the deployed CUDA provider for rows 1, 17 and 65. The same
# vector-eight rank-two kernel reads each matrix's actual leading stride.
RMS_THREADS = {512: 64, 1024: 128, 1536: 192, 4096: 512}
PARAMETERS = {
    "rms": [8] * 9 + [4] * 3,
    "add": [8] * 4 + [4] * 3 + [8],
    "qk": [8, 8, 8, 4, 4, 8, 8, 8, 4, 4, 4, 8, 8],
}


def read_images(directory):
    if directory is None:
        raise ValueError("GLM image preparation requires --glm-vision-kernels with the deployed CUDA images")
    directory = Path(directory)
    if json.loads((directory / "manifest.json").read_text()) != MANIFEST:
        raise ValueError("GLM vision kernel manifest differs from the validated deployment")
    images = {}
    for name, expected in MANIFEST["images"].items():
        image = (directory / name).read_bytes()
        if hashlib.sha256(image).hexdigest() != expected:
            raise ValueError(f"{name}: deployed GLM vision CUDA image digest changed")
        images[name] = image
    return images
