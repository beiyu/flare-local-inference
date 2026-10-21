"""GLM projections selected from their stored tensor representation."""

import torch

from ignition.components.linear import QuantizedProjection, UnquantizedProjection, BatchedQuantizedProjection
from ignition.preparation.tensor_metadata import record_numeric_tensor


def rms_norm(source, weight, output, epsilon, *, projection_weight, kernels=None):
    """Preserve the normalization arithmetic of the stored projection precision."""
    if isinstance(projection_weight, torch.Tensor):
        from ignition.components.glm_vision_kernels import original_rms_matrix_out

        if kernels is None:
            raise ValueError("native BF16 GLM normalization requires the deployed RMSNorm image")
        original_rms_matrix_out(source, weight, output, epsilon, kernels.handle)
        return output
    from flashinfer.triton.norm import rms_norm as normalize

    # Packed-weight packages retain their original normalization arithmetic.
    return normalize(source, weight, output, epsilon, round_before_weight=True)


def projection(source, weight, output, *, session, name, workspace):
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name)
    return QuantizedProjection(source, weight, output, session=session, name=name,
                               workspace=workspace, activation_mode="auto")


class BatchedBF16Projection:
    """Stored BF16 MLA key/value weights, with caller-owned strided head outputs.

    The existing native BMM capture primitive writes into strided head views on
    all NVIDIA architectures. Weights remain BF16; no MXFP8 packing is performed.
    """

    def __init__(self, source, weight, output, *, major, session, name):
        rhs = weight if major == "n" else weight.transpose(1, 2)
        if (major not in ("n", "k") or source.ndim != 3 or output.ndim != 3 or rhs.ndim != 3
                or rhs.shape != (source.shape[0], source.shape[2], output.shape[2])
                or source.shape[:2] != output.shape[:2]
                or any(t.dtype != torch.bfloat16 or t.device != source.device for t in (source, weight, output))):
            raise ValueError("BF16 MLA absorption requires matching head, row and channel geometry")
        self.weight, self.output = rhs, output
        record_numeric_tensor(weight)

    def __call__(self, source):
        torch.ops.flare.sm103_bmm_out(source, self.weight, self.output)
        return self.output


def batched_projection(source, weight, output, **kwargs):
    constructor = BatchedBF16Projection if isinstance(weight, torch.Tensor) else BatchedQuantizedProjection
    return constructor(source, weight, output, **kwargs)
