"""Preparation capture of the deployment's unchanged vision normalization kernels.

The existing VISION graph contract is unchanged. Python loads the fixed images
and records ordinary CUDA kernels; the native server replays the exported graph.
"""

import ctypes as C
import itertools
import math
import weakref

import torch

from ignition.preparation.glm53_vision_kernels import PARAMETERS, RMS_THREADS, SYMBOLS, read_images

_LIBRARIES = weakref.WeakValueDictionary()
_HANDLES = itertools.count()


class OriginalVisionKernels:
    def __init__(self, directory, device):
        if torch.cuda.get_device_capability(device) != (12, 1) or torch.cuda.is_current_stream_capturing():
            raise ValueError("deployed GLM vision kernels require uncaptured SM121 preparation")
        self.driver = C.CDLL("libcuda.so.1")
        self.modules, self.functions = [], {}
        try:
            for name, image in read_images(directory).items():
                module = C.c_void_p()
                self._call("cuModuleLoadData", C.byref(module), C.create_string_buffer(image))
                self.modules.append(module)
                for key in (("rms", "add") if name == "rms.cubin" else ("qk",)):
                    function = C.c_void_p()
                    self._call("cuModuleGetFunction", C.byref(function), module, SYMBOLS[key].encode())
                    for index, expected in enumerate(PARAMETERS[key]):
                        offset, size = C.c_size_t(), C.c_size_t()
                        self._call("cuFuncGetParamInfo", function, C.c_size_t(index), C.byref(offset), C.byref(size))
                        if size.value != expected:
                            raise ValueError(f"original vision {key} parameter {index} ABI changed")
                    offset, size = C.c_size_t(), C.c_size_t()
                    if self.driver.cuFuncGetParamInfo(function, C.c_size_t(len(PARAMETERS[key])), C.byref(offset), C.byref(size)) != 1:
                        raise ValueError(f"original vision {key} parameter count changed")
                    self.functions[key] = function
        except BaseException:
            self.close()
            raise
        self.handle = next(_HANDLES)
        _LIBRARIES[self.handle] = self

    def _call(self, name, *arguments):
        status = getattr(self.driver, name)(*arguments)
        if status:
            message = C.c_char_p()
            self.driver.cuGetErrorString(C.c_int(status), C.byref(message))
            raise RuntimeError(f"{name}: CUDA {status}: {message.value!r}")

    def launch(self, name, arguments, rows, stream, *, threads=None):
        parameters = (C.c_void_p * len(arguments))(*[C.addressof(value) for value in arguments])
        if threads is None:
            threads = 256 if name == "add" else 128
        self._call("cuLaunchKernel", self.functions[name], C.c_uint(rows), C.c_uint(2 if name == "qk" else 1),
            C.c_uint(1), C.c_uint(threads), C.c_uint(1), C.c_uint(1),
            C.c_uint(8 if name == "qk" else 0), C.c_void_p(stream), parameters, None)

    def close(self):
        while self.modules:
            self._call("cuModuleUnload", self.modules.pop())
        self.functions.clear()
        if hasattr(self, "handle"):
            _LIBRARIES.pop(self.handle, None)


def _pointer(tensor):
    return C.c_void_p(tensor.data_ptr())


@torch.library.custom_op("flare::glm_vision_original_rms_out", mutates_args=("output",))
def original_rms_out(source: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    arguments = [_pointer(output), _pointer(source), C.c_int64(1024), *[C.c_int64(0) for _ in range(4)],
        _pointer(weight), C.c_int64(0), C.c_float(1e-6), C.c_int(source.shape[0]), C.c_int(1024)]
    _LIBRARIES[handle].launch("rms", arguments, source.shape[0], torch.cuda.current_stream(source.device).cuda_stream)


@original_rms_out.register_fake
def _fake_rms(source, weight, output, handle):
    pass


@torch.library.custom_op("flare::glm_vision_original_add_rms", mutates_args=("source", "residual"))
def original_add_rms(source: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, handle: int) -> None:
    arguments = [_pointer(source), C.c_int64(1024), _pointer(residual), _pointer(weight), C.c_float(1e-6),
        C.c_int(source.shape[0]), C.c_int(1024), C.c_int64(1024)]
    _LIBRARIES[handle].launch("add", arguments, source.shape[0], torch.cuda.current_stream(source.device).cuda_stream)


@original_add_rms.register_fake
def _fake_add_rms(source, residual, weight, handle):
    pass


@torch.library.custom_op("flare::glm_vision_original_qk", mutates_args=("query", "key"))
def original_qk(query: torch.Tensor, key: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, handle: int) -> None:
    arguments = [_pointer(query), _pointer(query), _pointer(q_weight), C.c_int(64), C.c_int(64),
        _pointer(key), _pointer(key), _pointer(k_weight), C.c_int(64), C.c_int(64), C.c_float(1e-5),
        C.c_void_p(0), C.c_void_p(0)]
    _LIBRARIES[handle].launch("qk", arguments, query.numel() // 64, torch.cuda.current_stream(query.device).cuda_stream)


@original_qk.register_fake
def _fake_qk(query, key, q_weight, k_weight, handle):
    pass


def _rms_matrix_layout(source, weight, output, eps):
    if (source.ndim != 2 or source.shape[1] not in RMS_THREADS
            or not 0 < source.shape[0] < 2**31 or output.shape != source.shape
            or weight.shape != source.shape[1:]
            or any(value.dtype != torch.bfloat16 for value in (source, weight, output))
            or source.device.type != "cuda" or source.device != weight.device or source.device != output.device
            or source.stride(1) != 1 or source.stride(0) < source.shape[1] or source.stride(0) % 8
            or not output.is_contiguous() or not weight.is_contiguous()
            or not math.isfinite(eps) or eps <= 0):
        raise ValueError("original RMSNorm requires aligned CUDA BF16 matrix rows, a matching contiguous output/gain, and finite positive epsilon")
    return source.shape[1]


@torch.library.custom_op("flare::glm_original_rms_matrix_out", mutates_args=("output",))
def original_rms_matrix_out(source: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
                            eps: float, handle: int) -> None:
    width = _rms_matrix_layout(source, weight, output, eps)
    if any(value.data_ptr() % 16 for value in (source, weight, output)):
        raise ValueError("original RMSNorm vector-eight operands must be 16-byte aligned")
    arguments = [_pointer(output), _pointer(source), C.c_int64(source.stride(0)),
        *[C.c_int64(0) for _ in range(4)], _pointer(weight), C.c_int64(0), C.c_float(eps),
        C.c_int(source.shape[0]), C.c_int(width)]
    _LIBRARIES[handle].launch("rms", arguments, source.shape[0],
        torch.cuda.current_stream(source.device).cuda_stream, threads=RMS_THREADS[width])


@original_rms_matrix_out.register_fake
def _fake_rms_matrix(source, weight, output, eps, handle):
    _rms_matrix_layout(source, weight, output, eps)
