"""Extract the deployed EXL3 thin/E3 kernels without importing its Torch ABI.

Only the read-only original extension is accepted. Its exact SM121a images are
loaded by the preparation adapter and subsequently captured/exported normally.
The native runtime never loads the Python extension or executes this extractor.
"""

import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

EXTENSION_SHA256 = "7d6eb20ac755e4f793c3a04889581a21ed4e1c352b6d201556e4634bd2524f90"
IMAGES = {
    "exl3_moe_inst_4.sm_121a.cubin": "4f9068146bcf0ab9fe24e32630c2713a1d3677bb6a003e286c98e32a2eefc37d",
    "exl3_fat_moe.sm_121a.cubin": "7379dce729f9fe0f78788dcb30285cf3f2aaca90840b9f63ee1a91e078bacf7c",
}
SYMBOLS = {
    "thin": "_Z15exl3_moe_kernelILi4ELi256EEvPK6__halfPS0_S3_S3_S3_PfPPKtPS2_S8_S7_S8_S8_S7_S8_S8_PKlSA_S2_iiiiiifiiiiPi",
    "gather": "_ZN48_GLOBAL__N__de95e1e6_15_exl3_fat_moe_cu_c426306016fm_gather_kernelEPK6__halfPKlPKiPKS2_PS0_S6_i",
    "gateup": "_ZN48_GLOBAL__N__de95e1e6_15_exl3_fat_moe_cu_c426306016fm_gateup_kernelEPK6__halfPKPKtS6_PKS2_S8_S8_PS0_PKiSB_SB_SB_iif",
    "down": "_ZN48_GLOBAL__N__de95e1e6_15_exl3_fat_moe_cu_c426306014fm_down_kernelEPK6__halfPKPKtPKS2_PfPKlS2_PKiSD_SD_SD_ii",
}


def extract_images(extension):
    extension = Path(extension).resolve()
    digest = hashlib.sha256()
    with extension.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    if digest.hexdigest() != EXTENSION_SHA256:
        raise ValueError("EXL3 prefill extension differs from the deployed thin/E3 binary")
    executable = shutil.which("cuobjdump")
    if executable is None:
        raise ValueError("preparation requires CUDA cuobjdump to read the original EXL3 images")
    with tempfile.TemporaryDirectory(prefix="flare-exl3-prefill-") as temporary:
        subprocess.run([executable, "-xelf", "exl3_moe_inst_4,exl3_fat_moe", str(extension)],
            cwd=temporary, check=True, capture_output=True, text=True)
        images = {}
        for name, expected in IMAGES.items():
            image = (Path(temporary) / name).read_bytes()
            if hashlib.sha256(image).hexdigest() != expected:
                raise ValueError(f"{name}: original EXL3 CUDA image digest changed")
            images[name] = image
    return images
