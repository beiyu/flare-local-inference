"""Preparation rejects altered original vision images before loading CUDA code."""

import json

import pytest

from ignition.preparation.glm53_vision_kernels import MANIFEST, read_images


def test_missing_original_image_directory_is_explicit():
    with pytest.raises(ValueError, match="requires --glm-vision-kernels"):
        read_images(None)


def test_manifest_cannot_select_an_unvalidated_kernel(tmp_path):
    manifest = {**MANIFEST, "architecture": "sm120"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest differs"):
        read_images(tmp_path)


def test_modified_image_is_rejected_before_cuda_loading(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    (tmp_path / "rms.cubin").write_bytes(b"altered CUDA image")
    with pytest.raises(ValueError, match="rms.cubin:.*digest changed"):
        read_images(tmp_path)
