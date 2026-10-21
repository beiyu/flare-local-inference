"""Caller-owned capture of the deployment's EXL3 cooperative expert ABI.

Python loads the validated specialization and records launches during preparation.
Native serving replays the exported kernels and relocated pointer tables; it does
not load this library or call Python. Every row slice and layer shares one serial
workspace. Complete-forward compilation sees every read and mutation explicitly.
"""

import ctypes as C
import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
import weakref

import torch

from ignition.preparation.exl3 import (
    BITS, EXPERTS, GEOMETRY, HIDDEN, LIBRARY_SHA256, MAX_ROWS,
    POINTER_KEYS, TOPK, row_slices,
)
from ignition.preparation.tensor_metadata import record_numeric_tensor

_LIBRARIES = weakref.WeakValueDictionary()
_HANDLES = itertools.count()
_LOCAL_INTERMEDIATE = 1024
_SLOTS = MAX_ROWS * TOPK
_COUNTERS = _SLOTS * (_LOCAL_INTERMEDIATE // 128) + MAX_ROWS * (HIDDEN // 128) + 2 + 2 * _SLOTS + 1


class CooperativeLibrary:
    def __init__(self, path, device):
        path = Path(path)
        if torch.cuda.get_device_capability(device) != (12, 1):
            raise ValueError("the deployed EXL3 cooperative specialization requires SM121")
        if hashlib.sha256(path.read_bytes()).hexdigest() != LIBRARY_SHA256:
            raise ValueError("EXL3 cooperative binary differs from the validated GLM specialization")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EXL3 initialization must finish before graph capture")
        self.library = C.CDLL(str(path))
        self.library.glm53_coop_abi.restype = C.c_int
        if self.library.glm53_coop_abi() != 1:
            raise ValueError("unsupported EXL3 cooperative ABI")
        self.library.glm53_coop_info.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_int)]
        self.library.glm53_coop_info.restype = C.c_int
        info = (C.c_int * 18)()
        if self.library.glm53_coop_info(BITS, GEOMETRY, info):
            raise RuntimeError("EXL3 cooperative geometry preparation failed")
        if info[16] != _COUNTERS or info[17] != 344 or min(info[4], info[9], info[14]) < 1:
            raise ValueError("EXL3 cooperative storage or residency differs from the validated ABI")
        self.launch = self.library.glm53_coop_launch
        self.launch.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_int, C.c_int,
                                C.c_float, C.c_int, C.c_int, C.c_void_p]
        self.launch.restype = C.c_int
        self.handle = next(_HANDLES)
        _LIBRARIES[self.handle] = self


@torch.library.custom_op("flare::exl3_cooperative_out", mutates_args=("scratch", "counters", "output"))
def _cooperative_out(source: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                     tables: list[torch.Tensor], scratch: list[torch.Tensor],
                     counters: torch.Tensor, output: torch.Tensor, experts: int,
                     limit: float, handle: int) -> None:
    tensors = (source, ids, weights, *tables, *scratch, counters, output)
    pointers = (C.c_void_p * 20)(*[tensor.data_ptr() for tensor in tensors])
    status = _LIBRARIES[handle].launch(pointers, BITS, source.shape[0], experts, limit, GEOMETRY, 0,
        C.c_void_p(torch.cuda.current_stream(source.device).cuda_stream))
    if status:
        raise RuntimeError(f"EXL3 cooperative launch failed with CUDA status {status}")


@_cooperative_out.register_fake
def _fake_cooperative_out(source, ids, weights, tables, scratch, counters, output, experts, limit, handle):
    pass


class Exl3Workspace:
    """One rank's serial workspace, shared by all layers and graph capacities."""

    def __init__(self, library, device, *, prefill_library=None, prefill_capacity=4096):
        self.native = CooperativeLibrary(library, device)
        self.source = torch.empty((MAX_ROWS, HIDDEN), dtype=torch.float16, device=device)
        self.ids = torch.zeros((MAX_ROWS, TOPK), dtype=torch.int64, device=device)
        self.weights = torch.empty((MAX_ROWS, TOPK), dtype=torch.float16, device=device)
        self.output = torch.empty((MAX_ROWS, HIDDEN), dtype=torch.float32, device=device)
        self.short_output = torch.empty((MAX_ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        self.scratch = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in (
            ((_SLOTS, HIDDEN), torch.float16), ((_SLOTS, HIDDEN), torch.float16),
            ((_SLOTS, _LOCAL_INTERMEDIATE), torch.float16), ((_SLOTS, _LOCAL_INTERMEDIATE), torch.float16),
            ((_SLOTS, _LOCAL_INTERMEDIATE), torch.float16), ((_SLOTS, HIDDEN), torch.float32))]
        self.counters = torch.zeros(_COUNTERS, dtype=torch.int32, device=device)
        self.prefill = None
        if prefill_library is not None:
            from .exl3_prefill import PrefillWorkspace

            self.prefill = PrefillWorkspace(prefill_library, prefill_capacity, device)

    def tensors(self):
        return (self.source, self.ids, self.weights, self.output, self.short_output, *self.scratch, self.counters,
                *(() if self.prefill is None else self.prefill.tensors()))

    def close(self):
        if self.prefill is not None:
            self.prefill.close()


@dataclass(frozen=True)
class Exl3Experts:
    tensors: tuple[dict[str, torch.Tensor], ...]
    tables: tuple[torch.Tensor, ...]
    workspace: Exl3Workspace
    hidden_size: int = HIDDEN
    intermediate_size: int = _LOCAL_INTERMEDIATE

    @property
    def num_experts(self):
        return len(self.tensors)

    @classmethod
    def from_tensors(cls, tensors, workspace):
        tensors = tuple(tensors)
        if not 0 < len(tensors) <= EXPERTS:
            raise ValueError("EXL3 pointer tables require 1 through 288 experts")
        device = workspace.source.device
        for expert in tensors:
            if set(expert) != set(POINTER_KEYS):
                raise ValueError("EXL3 expert must have all nine trellis/scale planes")
            for key, tensor in expert.items():
                projection, suffix = key.split(".")
                inputs, outputs = ((HIDDEN, _LOCAL_INTERMEDIATE)
                    if projection != "down_proj" else (_LOCAL_INTERMEDIATE, HIDDEN))
                expected = ((inputs // 16, outputs // 16, BITS * 16) if suffix == "trellis"
                            else (inputs,) if suffix == "suh" else (outputs,))
                dtype = torch.int16 if suffix == "trellis" else torch.float16
                if (tensor.shape != expected or tensor.dtype != dtype or tensor.device != device
                        or not tensor.is_contiguous()):
                    raise ValueError(f"EXL3 {key} must preserve the TP2 checkpoint geometry and dtype")
                record_numeric_tensor(tensor)
        # Pointer-bearing tables intentionally remain opaque to the exporter.
        tables = tuple(torch.tensor([expert[key].data_ptr() for expert in tensors], dtype=torch.int64,
                                    device=device) for key in POINTER_KEYS)
        return cls(tensors, tables, workspace)


class PreparedCooperativeExl3Experts:
    """All physical rows use the original K4 MCG arithmetic in fixed <=32-row slices.

    This also covers prefill without the deployment's dynamic fat-expert dispatch.
    It is a prepared graph composition, and needs full numerical qualification
    against that dispatch before the complete model can be called compatible.
    """

    def __init__(self, source, experts, ids, weights, output, *, limit=10.):
        rows = source.shape[0]
        if (source.shape != (rows, HIDDEN) or source.dtype != torch.bfloat16
                or output.shape != source.shape or output.dtype != source.dtype
                or ids.shape != (rows, TOPK) or ids.dtype != torch.int32
                or weights.shape != ids.shape or weights.dtype != torch.float32
                or any(t.device != source.device or not t.is_contiguous() for t in (source, output, ids, weights))
                or source.device != experts.workspace.source.device or limit != 10.):
            raise ValueError("EXL3 execution requires fixed BF16 rows, native route storage and SwiGLU limit 10")
        self.source, self.ids, self.weights, self.output, self.experts = source, ids, weights, output, experts
        self.limit = float(limit)
        self.slices = row_slices(rows)

    def __call__(self):
        workspace = self.experts.workspace
        for first, last in self.slices:
            rows = last - first
            source, ids, weights, output = (workspace.source[:rows], workspace.ids[:rows],
                workspace.weights[:rows], workspace.output[:rows])
            source.copy_(self.source[first:last])
            selected = self.ids[first:last]
            ids.copy_(torch.where((selected < 0) | (selected >= self.experts.num_experts),
                                  self.experts.num_experts, selected))
            weights.copy_(self.weights[first:last])
            _cooperative_out(source, ids, weights, list(self.experts.tables), workspace.scratch,
                workspace.counters, output, self.experts.num_experts, self.limit, workspace.native.handle)
            # The deployed apply_exl3_experts also rounds its FP32 sum to the input dtype before shared addition.
            self.output[first:last].copy_(output)
        return self.output


class PreparedExl3Experts(PreparedCooperativeExl3Experts):
    """Preserve the deployed physical-row dispatch, including short prefill.

    The original wrapper receives 1..32 actual rows for cooperative calls.
    Padded captures use the GPU valid-token count to select that same arithmetic;
    larger valid batches retain the original thin/E3 32-row expert threshold.
    """

    def __init__(self, source, experts, ids, weights, output, *, limit=10., num_valid_tokens=None):
        super().__init__(source, experts, ids, weights, output, limit=limit)
        self.prefill = self.short = None
        self.num_valid_tokens = num_valid_tokens
        if num_valid_tokens is not None and (num_valid_tokens.shape != (1,)
                or num_valid_tokens.dtype != torch.int32 or num_valid_tokens.device != source.device):
            raise ValueError("EXL3 valid-token count must be a device int32 scalar")
        if source.shape[0] > MAX_ROWS:
            from .exl3_prefill import PreparedExl3PrefillExperts

            self.prefill = PreparedExl3PrefillExperts(source, experts, ids, weights, output)
            if num_valid_tokens is not None:
                self.short = PreparedCooperativeExl3Experts(source[:MAX_ROWS], experts,
                    ids[:MAX_ROWS], weights[:MAX_ROWS], experts.workspace.short_output, limit=limit)

    def __call__(self):
        if self.prefill is None:
            return super().__call__()
        self.prefill()
        if self.short is not None:
            short = self.short()
            self.output[:MAX_ROWS].copy_(torch.where(self.num_valid_tokens <= MAX_ROWS,
                short, self.output[:MAX_ROWS]))
        return self.output
