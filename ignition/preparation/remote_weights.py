"""Adapt b12x routed reads to authenticated remote or pinned local sources."""

from array import array
import os
from pathlib import Path
import tempfile

from b12x.loader._checkpoint import DirectWeightSession, TensorRange

from .checkpoint_source import BLOCK_BYTES, file_identity, row_extent, tensor_dtype


def chunks(offset, width, rows, source_stride, destination, destination_stride, expand):
    """Partition rows without transferring source gaps or changing destination offsets."""
    if any(type(value) is not int or value < 0 or value >= 1 << 63
           for value in (offset, width, rows, source_stride, destination, destination_stride)):
        raise ValueError("invalid routed checkpoint geometry")
    if rows < 1 or expand not in (0, 1) or (expand and width % 2):
        raise ValueError("invalid routed checkpoint expansion")
    if (offset + (rows - 1) * source_stride + width >= 1 << 63 or
            destination + (rows - 1) * destination_stride + width * (1 + expand) >= 1 << 63):
        raise ValueError("routed checkpoint extent overflows signed bounds")
    if rows > 1 and destination_stride < width * (1 + expand):
        raise ValueError("routed checkpoint destinations overlap")
    for column in range(0, width, BLOCK_BYTES):
        take = min(BLOCK_BYTES, width - column)
        count = 0
        while count < rows:
            capacity = min(BLOCK_BYTES // take,
                           (BLOCK_BYTES - take) // source_stride + 1 if source_stride else rows)
            batch = min(rows - count, capacity)
            yield (offset + count * source_stride + column, take, batch, source_stride,
                   destination + count * destination_stride + column * (1 + expand), destination_stride)
            count += batch


def destinations_overlap(left, right):
    """Intersect routed row bytes without treating strided holes as writes."""
    a, aw, ac, ast = left
    b, bw, bc, bst = right
    i = j = 0
    while i < ac and j < bc:
        start_a, start_b = a + i * ast, b + j * bst
        if start_a + aw <= start_b:
            if not ast:
                return False
            i = max(i + 1, (start_b - a - aw) // ast + 1)
        elif start_b + bw <= start_a:
            if not bst:
                return False
            j = max(j + 1, (start_a - b - bw) // bst + 1)
        else:
            return True
    return False


class RemoteWeightSession(DirectWeightSession):
    """Keep b12x routing and native copies under the authenticated source owner.

    Routed source IDs refer to admitted names. Local mirrors use inode-pinned
    O_DIRECT descriptors; remote reads use a private bounded transfer buffer.
    """

    def __init__(self, source, manifest, *, device):
        if source.local is not None:
            from ignition.components.weights import _checkpoint_read_mode

            paths = (Path(f"/proc/self/fd/{source.local.files[source.name(manifest.snapshot / shard.path)].descriptor}")
                     for shard in manifest.shards)
            super().__init__(device=device, read_mode=_checkpoint_read_mode(paths))
        else:
            super().__init__(device=device, io_threads=1, read_mode="bounce")
        self.source, self.checkpoint = source, manifest
        self.local_files = {}
        self.buffer_directory = self.buffer_write = self.buffer_read = None
        try:
            self.names = tuple(source.name(manifest.snapshot / shard.path) for shard in manifest.shards)
            self.ids = {name: index for index, name in enumerate(self.names)}
            self.shards = {name: shard.path for name, shard in zip(self.names, manifest.shards, strict=True)}
            if source.local is not None:
                for name in self.names:
                    held = source.local.files[name]
                    held.validate()
                    descriptor = os.open(f"/proc/self/fd/{held.descriptor}",
                                         os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC)
                    self.local_files[name] = descriptor
                    if file_identity(descriptor) != held.identity:
                        raise RuntimeError("local checkpoint descriptor differs from its admitted inode")
                    held.validate()
            else:
                self.buffer_directory = tempfile.TemporaryDirectory(prefix="read-", dir=source.temporary.name)
                path = Path(self.buffer_directory.name) / "buffer"
                self.buffer_write = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
                self.buffer_read = os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC)
        except BaseException:
            self._close()
            raise

    def manifest(self, path):
        name = self.source.name(path)
        identifier = self.ids[name]
        return [TensorRange(tensor.name, identifier, tensor.offset, tensor.nbytes,
                            tensor.shape, tensor_dtype(tensor.dtype))
                for tensor in self.checkpoint.tensors.values() if tensor.shard == self.shards[name]]

    def _metadata_values(self, entries):
        import math
        import torch

        values = {}
        for entry in entries:
            self.metadata_bytes += entry.nbytes
            if self.metadata_bytes > 64 << 20:
                raise ValueError("declared metadata spans exceed the 64 MiB session budget")
            content = bytearray()
            for offset in range(0, entry.nbytes, BLOCK_BYTES):
                content.extend(self.source.read(self.names[entry.fd], entry.offset + offset,
                                                min(BLOCK_BYTES, entry.nbytes - offset)))
            values[entry.name] = (torch.frombuffer(content, dtype=entry.dtype, count=math.prod(entry.shape))
                                  .reshape(entry.shape) if entry.nbytes
                                  else torch.empty(entry.shape, dtype=entry.dtype))
        return values

    def _execute(self, records):
        if len(records) % 8:
            raise ValueError("invalid b12x checkpoint descriptor size")
        destinations = []
        for index in range(0, len(records), 8):
            identifier, offset, width, pointer, transform, rows, source_stride, stride = records[index:index + 8]
            if transform not in (0, 1, 2) or rows < 1:
                raise ValueError("unsupported b12x checkpoint transform")
            if width:
                written = width * (1 + (transform == 1))
                if rows > 1 and stride < written:
                    raise ValueError("routed checkpoint destinations overlap")
                destinations.append((pointer, written, rows, stride))
            if transform != 2:
                if identifier >= len(self.names):
                    raise ValueError("unknown remote checkpoint source ID")
                size = self.source.files[self.names[identifier]]["bytes"]
                if offset + (rows - 1) * source_stride + width > size:
                    raise ValueError("routed checkpoint view exceeds its source file")
        destinations.sort()
        for index, previous in enumerate(destinations):
            end = previous[0] + previous[1] + (previous[2] - 1) * previous[3]
            for following in destinations[index + 1:]:
                if following[0] >= end:
                    break
                if destinations_overlap(previous, following):
                    raise ValueError("overlapping checkpoint destinations require an explicit dependency")
        if self.source.local is not None:
            local_records = array("Q", records)
            involved = set()
            for index in range(0, len(records), 8):
                if records[index + 4] != 2:
                    values = records[index:index + 8]
                    _, offset, width, pointer, transform, rows, source_stride, stride = values
                    if (any(type(value) is not int or not 0 <= value < 1 << 63
                            for value in (offset, width, pointer, rows, source_stride, stride))
                            or (transform == 1 and width % 2)
                            or offset + (rows - 1) * source_stride + width >= 1 << 63
                            or pointer + (rows - 1) * stride + width * (1 + transform) >= 1 << 63):
                        raise ValueError("invalid routed checkpoint geometry or expansion")
                    name = self.names[records[index]]
                    involved.add(name)
                    local_records[index] = self.local_files[name]
            for name in involved:
                self.source.local.files[name].validate()
            try:
                # Preserve the complete native batch, including owned CPU scalar records.
                super()._execute(local_records)
            finally:
                for name in involved:
                    self.source.local.files[name].validate()
            for index in range(0, len(records), 8):
                if records[index + 4] != 2:
                    name = self.names[records[index]]
                    entry = self.source.files[name]
                    key = ("payload_bytes" if entry["header_bytes"] is not None
                           and records[index + 1] >= entry["header_bytes"] else "metadata_bytes")
                    self.source.local_stats[key] += records[index + 2] * records[index + 5]
                    self.source.local_stats["requests"] += 1
            return
        for index in range(0, len(records), 8):
            identifier, offset, width, pointer, transform, rows, source_stride, destination_stride = records[index:index + 8]
            if transform == 2:
                # These records name owned CPU scalar storage, not a checkpoint source ID.
                super()._execute(array("Q", records[index:index + 8]))
                continue
            name = self.names[identifier]
            for start, count, batch, stride, target, target_stride in chunks(
                    offset, width, rows, source_stride, pointer, destination_stride, transform):
                row_extent(self.source.files[name]["bytes"], start, count, batch, stride)
                content = self.source.read(name, start, count, batch, stride)
                os.ftruncate(self.buffer_write, len(content))
                written = 0
                while written < len(content):
                    size = os.pwrite(self.buffer_write, content[written:], written)
                    if size <= 0:
                        raise OSError("checkpoint transfer buffer write did not advance")
                    written += size
                # The synchronous native reader owns all destination and buffer access until it returns.
                super()._execute(array("Q", (self.buffer_read, 0, count, target, transform,
                                             batch, count, target_stride)))

    def stats(self, *, flush=True):
        result = super().stats(flush=flush)
        result["remote_source"] = dict(self.source.stats)
        if self.source.local is not None:
            result["local_source"] = dict(self.source.local_stats)
        return result

    def _close(self):
        try:
            super()._close()
        finally:
            for descriptor in self.local_files.values():
                os.close(descriptor)
            self.local_files.clear()
            for name in ("buffer_read", "buffer_write"):
                descriptor = getattr(self, name)
                if descriptor is not None:
                    setattr(self, name, None)
                    os.close(descriptor)
            if self.buffer_directory is not None:
                self.buffer_directory.cleanup()
                self.buffer_directory = None
