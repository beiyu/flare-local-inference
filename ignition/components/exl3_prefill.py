"""Prepared capture of the deployment's original thin plus E3 CUDA kernels."""

import ctypes as C
import itertools
import weakref

import torch

from ignition.preparation.exl3_prefill import SYMBOLS, extract_images
from ignition.preparation.tensor_metadata import record_numeric_tensor

_LIBRARIES = weakref.WeakValueDictionary()
_HANDLES = itertools.count()
_CAP, _HIDDEN, _INTERMEDIATE, _TOPK, _TILE = 32, 4096, 1024, 8, 64


class PrefillLibrary:
    """Own the exact original images until all captured graph work is drained."""

    def __init__(self, extension, device):
        if torch.cuda.get_device_capability(device) != (12, 1) or torch.cuda.is_current_stream_capturing():
            raise ValueError("EXL3 prefill initialization requires uncaptured SM121 preparation")
        self.driver = C.CDLL("libcuda.so.1")
        self.modules, self.functions = [], {}
        try:
            for name, image in extract_images(extension).items():
                module = C.c_void_p()
                self._call("cuModuleLoadData", C.byref(module), C.create_string_buffer(image))
                self.modules.append(module)
                for key in (("thin",) if "inst_4" in name else ("gather", "gateup", "down")):
                    function = C.c_void_p()
                    self._call("cuModuleGetFunction", C.byref(function), module, SYMBOLS[key].encode())
                    sizes = ({"thin": [8] * 18 + [4] * 11 + [8], "gather": [8] * 6 + [4],
                              "gateup": [8] * 11 + [4] * 3, "down": [8] * 10 + [4] * 2})[key]
                    for index, expected in enumerate(sizes):
                        offset, size = C.c_size_t(), C.c_size_t()
                        self._call("cuFuncGetParamInfo", function, C.c_size_t(index), C.byref(offset), C.byref(size))
                        if size.value != expected:
                            raise ValueError(f"original EXL3 {key} parameter {index} ABI changed")
                    offset, size = C.c_size_t(), C.c_size_t()
                    if self.driver.cuFuncGetParamInfo(function, C.c_size_t(len(sizes)), C.byref(offset), C.byref(size)) != 1:
                        raise ValueError(f"original EXL3 {key} parameter count changed")
                    if key in ("thin", "gateup", "down"):
                        self._call("cuFuncSetAttribute", function, C.c_int(8), C.c_int(92160 if key == "thin" else 32768))
                    self.functions[key] = function
        except BaseException:
            self.close()
            raise
        self.concurrency = torch.cuda.get_device_properties(device).multi_processor_count // 8
        if self.concurrency < 1:
            self.close()
            raise ValueError("original thin experts require at least eight SMs")
        self.handle = next(_HANDLES)
        _LIBRARIES[self.handle] = self

    def _call(self, name, *arguments):
        status = getattr(self.driver, name)(*arguments)
        if status:
            message = C.c_char_p()
            self.driver.cuGetErrorString(C.c_int(status), C.byref(message))
            raise RuntimeError(f"{name}: CUDA {status}: {message.value!r}")

    def launch(self, name, pointers, scalars, grid, threads, shared, stream):
        arguments = [C.c_void_p(tensor.data_ptr()) for tensor in pointers] + list(scalars)
        params = (C.c_void_p * len(arguments))(*[C.addressof(argument) for argument in arguments])
        self._call("cuLaunchKernel", self.functions[name], *[C.c_uint(value) for value in grid],
            C.c_uint(threads), C.c_uint(1), C.c_uint(1), C.c_uint(shared), C.c_void_p(stream), params, None)

    def close(self):
        while self.modules:
            self._call("cuModuleUnload", self.modules.pop())
        self.functions.clear()
        if hasattr(self, "handle"):
            _LIBRARIES.pop(self.handle, None)


@torch.library.custom_op("flare::exl3_original_prefill_out", mutates_args=("temps", "locks", "h13", "h2", "output"))
def _prefill_out(source: torch.Tensor, tables: list[torch.Tensor], counts: torch.Tensor,
                 token_sorted: torch.Tensor, weight_sorted: torch.Tensor, temps: list[torch.Tensor],
                 locks: torch.Tensor, h13: torch.Tensor, h2: torch.Tensor,
                 segments: list[torch.Tensor], row_expert: torch.Tensor, row_token: torch.Tensor,
                 row_weight: torch.Tensor, num_rows: torch.Tensor, num_segs: torch.Tensor,
                 output: torch.Tensor, experts: int, handle: int) -> None:
    native = _LIBRARIES[handle]
    stream = torch.cuda.current_stream(source.device).cuda_stream
    # Pointer/scalar order and launch geometry exactly match original exl3_moe.cu.
    native.launch("thin", [source, *temps, output, *tables, counts, token_sorted, weight_sorted],
        [C.c_int(v) for v in (_HIDDEN, _INTERMEDIATE, experts, _TOPK, _CAP, native.concurrency)]
            + [C.c_float(10.)] + [C.c_int(v) for v in (0, 4, 4, 4)] + [C.c_void_p(locks.data_ptr())],
        (8, 1, native.concurrency), 512, 92160, stream)
    rows_cap = h13.shape[0]
    grid_y = min(max(segments[0].numel(), 1), 512)
    native.launch("gather", [source, row_token, row_expert, tables[1], h13, num_rows], [C.c_int(_HIDDEN)],
        (min(max((rows_cap + 7) // 8, 1), 1024), _HIDDEN // 128, 1), 256, 0, stream)
    native.launch("gateup", [h13, tables[0], tables[3], tables[2], tables[5], tables[7], h2, *segments, num_segs],
        [C.c_int(_HIDDEN), C.c_int(_INTERMEDIATE), C.c_float(10.)], (_INTERMEDIATE // 128, grid_y, 1), 256, 32768, stream)
    native.launch("down", [h2, tables[6], tables[8], output, row_token, row_weight, *segments, num_segs],
        [C.c_int(_INTERMEDIATE), C.c_int(_HIDDEN)], (_HIDDEN // 256, grid_y, 1), 256, 32768, stream)


@_prefill_out.register_fake
def _fake_prefill_out(source, tables, counts, token_sorted, weight_sorted, temps, locks, h13, h2,
                      segments, row_expert, row_token, row_weight, num_rows, num_segs, output, experts, handle):
    pass


class PrefillWorkspace:
    """One maximum-capacity serial workspace per rank, shared by all layer variants."""

    def __init__(self, extension, capacity, device):
        if type(capacity) is not int or not _CAP < capacity < 2**31 // _TOPK:
            raise ValueError("EXL3 prefill capacity must exceed 32 and fit signed route indices")
        self.native = PrefillLibrary(extension, device)
        self.capacity = capacity
        slots = capacity * _TOPK
        self.source = torch.empty((capacity, _HIDDEN), dtype=torch.float16, device=device)
        self.output = torch.empty((capacity, _HIDDEN), dtype=torch.float32, device=device)
        self.h13 = torch.empty((slots, _HIDDEN), dtype=torch.float16, device=device)
        self.h2 = torch.empty((slots, _INTERMEDIATE), dtype=torch.float16, device=device)
        self.temps = [torch.empty((self.native.concurrency, _CAP, width), dtype=torch.float16, device=device)
                      for width in (_HIDDEN, _HIDDEN, _INTERMEDIATE, _INTERMEDIATE)]
        self.locks = torch.zeros(1024 * 1024 + 1024 * 2, dtype=torch.int32, device=device)
        self.counts = torch.zeros(289, dtype=torch.int64, device=device)

    def tensors(self):
        return (self.source, self.output, self.h13, self.h2, *self.temps, self.locks, self.counts)

    def close(self):
        self.native.close()


class PreparedExl3PrefillExperts:
    """Original device-side routing tables and unchanged thin/E3 kernel arithmetic."""

    def __init__(self, source, experts, ids, weights, output):
        rows = source.shape[0]
        self.source, self.ids, self.weights, self.output, self.experts = source, ids, weights, output, experts
        workspace = experts.workspace.prefill
        if workspace is None or not _CAP < rows <= workspace.capacity:
            raise ValueError("EXL3 rows above 32 require the original prefill extension and sufficient prepared capacity")
        self.workspace = workspace
        self.rows, self.slots = rows, rows * _TOPK
        self.expert_count = workspace.counts[:experts.num_experts + 1]
        self.flat_token = torch.arange(rows, dtype=torch.int64, device=source.device).repeat_interleave(_TOPK)
        self.ones = torch.ones(self.slots, dtype=torch.int64, device=source.device)
        self.row_indices = torch.arange(self.slots, device=source.device)
        max_segments = (self.slots + _TILE - 1) // _TILE + experts.num_experts
        self.segment_indices = torch.arange(max_segments, device=source.device)
        for tensor in (self.flat_token, self.ones, self.row_indices, self.segment_indices):
            record_numeric_tensor(tensor)

    def __call__(self):
        experts, workspace = self.experts, self.workspace
        source, output = workspace.source[:self.rows], workspace.output[:self.rows]
        source.copy_(self.source)
        output.zero_()
        ids = self.ids.reshape(-1).to(torch.int64)
        local = torch.where((ids < 0) | (ids >= experts.num_experts), experts.num_experts, ids)
        order = local.argsort()
        token_sorted = self.flat_token[order]
        weight_sorted = self.weights.reshape(-1).to(torch.float16)[order]
        self.expert_count.zero_()
        self.expert_count.scatter_add_(0, local, self.ones)
        counts = self.expert_count[:-1]
        segments, row_expert, row_token, row_weight, num_rows, num_segs = grouped_tables(
            counts, token_sorted, weight_sorted, self.row_indices, self.segment_indices)
        _prefill_out(source, list(experts.tables), self.expert_count, token_sorted, weight_sorted,
            workspace.temps, workspace.locks, workspace.h13[:self.slots], workspace.h2[:self.slots],
            segments, row_expert, row_token, row_weight, num_rows, num_segs, output,
            experts.num_experts, workspace.native.handle)
        self.output.copy_(output)
        return self.output


def grouped_tables(counts, token_sorted, weight_sorted, row_indices, segment_indices):
    """Original E3 capacity tables; only counts above 32 enter a fat segment."""
    fat_rows = torch.where(counts > _CAP, counts, 0)
    row_cum = torch.cumsum(fat_rows, 0)
    row_off = row_cum - fat_rows
    sorted_off = torch.cumsum(counts, 0) - counts
    tiles = (fat_rows + (_TILE - 1)) // _TILE
    tile_cum = torch.cumsum(tiles, 0)
    tile_off = tile_cum - tiles
    e = torch.searchsorted(tile_cum, segment_indices, right=True).clamp(max=counts.numel() - 1)
    local_tile = segment_indices - tile_off[e]
    segments = [e.to(torch.int32), (row_off[e] + local_tile * _TILE).to(torch.int32),
        (fat_rows[e] - local_tile * _TILE).clamp(min=0, max=_TILE).to(torch.int32)]
    re = torch.searchsorted(row_cum, row_indices, right=True).clamp(max=counts.numel() - 1)
    src = (sorted_off[re] + (row_indices - row_off[re])).clamp(max=token_sorted.numel() - 1)
    return (segments, re.to(torch.int32), token_sorted.index_select(0, src), weight_sorted.index_select(0, src),
            row_cum[-1:].to(torch.int32), tile_cum[-1:].to(torch.int32))
