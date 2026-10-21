"""CPU adapter qualification using real Linux O_DIRECT descriptors, without CUDA."""

from array import array
from contextlib import contextmanager
import ctypes
import fcntl
import json
import mmap
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from ignition.preparation import checkpoint_source as transport
from ignition.preparation.checkpoint import IndexedCheckpoint, ShardMetadata, TensorMetadata
from ignition.preparation.remote_weights import RemoteWeightSession


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux pinned O_DIRECT source contract")


def direct_bytes(descriptor, offset, size):
    """Exercise the actual O_DIRECT descriptor with aligned CPU read storage."""
    page = mmap.PAGESIZE
    start = offset // page * page
    extent = ((offset + size - start + page - 1) // page) * page
    with mmap.mmap(-1, extent) as buffer:
        count = os.preadv(descriptor, [buffer], start)
        assert count >= offset - start + size
        return buffer[offset - start:offset - start + size]


@contextmanager
def local_session(tmp_path, *, execute=None):
    original = tmp_path / "original"
    original.mkdir()
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    tensors, shards, payloads = {}, [], {}
    for index, name in enumerate(("a.safetensors", "b.safetensors")):
        payload = bytes((value + index * 97) % 256 for value in range(8192))
        tensor = f"tensor{index}"
        header = json.dumps({tensor: {"dtype": "U8", "shape": [8192],
                                     "data_offsets": [0, 8192]}}).encode()
        content = struct.pack("<Q", len(header)) + header + payload
        target = original / f"payload{index}"
        target.write_bytes(content)
        (original / name).symlink_to(target.name)
        # RootSource catalogs all files. Keep underlying payloads outside the snapshot.
        outside = tmp_path / target.name
        target.rename(outside)
        (original / name).unlink()
        (original / name).symlink_to(outside)
        offset = 8 + len(header)
        tensors[tensor] = TensorMetadata(tensor, name, offset, len(payload), (8192,), "U8")
        shards.append(ShardMetadata(name, len(content), offset, "", 1))
        payloads[name] = payload
    local = transport.RootSource(original, SimpleNamespace(repository="test", revision="pinned"))
    source = SimpleNamespace(local=local, snapshot=metadata, files={entry["path"]: entry for entry in local.catalog["files"]},
                             stats=dict(payload_bytes=0, metadata_bytes=0, requests=0),
                             local_stats=dict(payload_bytes=0, metadata_bytes=0, requests=0))
    source.name = lambda path: Path(path).relative_to(metadata).as_posix()
    source.read = lambda *args: local.read(*args)
    manifest = IndexedCheckpoint(metadata, tensors, tuple(shards))
    calls = []

    def init(self, **options):
        self.metadata_bytes = 0
        self.options = options

    def native(self, records):
        calls.append(array("Q", records))
        if execute is not None:
            execute(self, records)

    session = None
    try:
        with mock.patch("b12x.loader._checkpoint.DirectWeightSession.__init__", init), \
                mock.patch("b12x.loader._checkpoint.DirectWeightSession._execute", native), \
                mock.patch("b12x.loader._checkpoint.DirectWeightSession._close"):
            session = RemoteWeightSession(source, manifest, device=0)
            yield session, source, calls, payloads
            session._close()
            session = None
    finally:
        if session is not None:
            with mock.patch("b12x.loader._checkpoint.DirectWeightSession._close"):
                session._close()
        local.close()


def test_local_batch_preserves_records_cpu_scalars_and_original_inodes(tmp_path):
    observed = []
    scalar = ctypes.create_string_buffer(b"CPU!")

    def execute(session, records):
        for index in range(0, len(records), 8):
            descriptor, offset, width, _, transform, rows, source_stride, _ = records[index:index + 8]
            observed.append(ctypes.string_at(offset, width) if transform == 2 else b"".join(
                direct_bytes(descriptor, offset + row * source_stride, width) for row in range(rows)))

    with local_session(tmp_path, execute=execute) as (session, source, calls, payloads):
        a, b = [session.checkpoint.tensors[f"tensor{i}"] for i in range(2)]
        records = array("Q", (0, a.offset + 1, 2, 1000, 0, 3, 8, 8,
                              1, b.offset + 4, 4, 2000, 1, 2, 12, 16,
                              0, ctypes.addressof(scalar), 4, 3000, 2, 1, 0, 0))
        original = array("Q", records)
        with mock.patch("ignition.preparation.remote_weights.tempfile.TemporaryDirectory", side_effect=AssertionError), \
                mock.patch("ignition.preparation.remote_weights.os.pwrite", side_effect=AssertionError), \
                mock.patch.object(source, "read", side_effect=AssertionError):
            session._execute(records)
        assert records == original
        assert len(calls) == 1
        expected = array("Q", records)
        for index, name in ((0, "a.safetensors"), (8, "b.safetensors")):
            descriptor = session.local_files[name]
            expected[index] = descriptor
            assert transport.file_identity(descriptor) == source.local.files[name].identity
            assert fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_DIRECT
            assert not os.get_inheritable(descriptor)
        assert calls[0] == expected
        assert observed == [b"".join(payloads["a.safetensors"][start:start + 2] for start in (1, 9, 17)),
                            b"".join(payloads["b.safetensors"][start:start + 4] for start in (4, 16)), b"CPU!"]
        assert session.buffer_directory is session.buffer_read is session.buffer_write is None
        assert "io_threads" not in session.options  # Native local-session default, without the remote single-buffer limit.
        assert source.stats == dict(payload_bytes=0, metadata_bytes=0, requests=0)
        assert source.local_stats == dict(payload_bytes=14, metadata_bytes=0, requests=2)
        descriptors = tuple(session.local_files.values())
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_local_metadata_uses_source_ids_and_held_reader(tmp_path):
    import torch

    with local_session(tmp_path) as (session, source, calls, payloads):
        entries = session.manifest(source.snapshot / "b.safetensors")
        assert len(entries) == 1 and entries[0].fd == 1
        values = session._metadata_values(entries)
        assert values["tensor1"].dtype == torch.uint8
        assert bytes(values["tensor1"].tolist()) == payloads["b.safetensors"]
        assert session.metadata_bytes == 8192
        assert calls == []
        assert source.stats == dict(payload_bytes=0, metadata_bytes=0, requests=0)


@pytest.mark.parametrize("mutation", ["retarget", "payload"])
def test_local_changes_before_execution_are_rejected(tmp_path, mutation):
    with local_session(tmp_path) as (session, source, calls, _):
        held = source.local.files["a.safetensors"]
        if mutation == "retarget":
            held.path.unlink()
            held.path.symlink_to(source.local.files["b.safetensors"].path)
        else:
            with held.path.open("r+b") as stream:
                stream.seek(session.checkpoint.tensors["tensor0"].offset)
                stream.write(b"changed")
        with pytest.raises(RuntimeError, match="changed"):
            session._execute(array("Q", (0, 0, 8, 1000, 0, 1, 0, 0)))
        assert calls == []


def test_transient_symlink_retarget_during_native_read_cannot_substitute_payload(tmp_path):
    observed = []

    def execute(session, records):
        held = session.source.local.files["a.safetensors"]
        target = held.path.readlink()
        held.path.unlink()
        held.path.symlink_to(session.source.local.files["b.safetensors"].path)
        try:
            observed.append(direct_bytes(records[0], records[1], records[2]))
        finally:
            held.path.unlink()
            held.path.symlink_to(target)

    with local_session(tmp_path, execute=execute) as (session, _, calls, payloads):
        offset = session.checkpoint.tensors["tensor0"].offset
        session._execute(array("Q", (0, offset, 16, 1000, 0, 1, 0, 0)))
        assert len(calls) == 1
        assert observed == [payloads["a.safetensors"][:16]]
        assert observed != [payloads["b.safetensors"][:16]]


def test_mutation_during_native_read_fails_post_validation(tmp_path):
    def execute(session, records):
        with session.source.local.files["a.safetensors"].path.open("r+b") as stream:
            stream.seek(records[1])
            stream.write(b"changed")

    with local_session(tmp_path, execute=execute) as (session, source, calls, _):
        with pytest.raises(RuntimeError, match="changed"):
            session._execute(array("Q", (0, 0, 8, 1000, 0, 1, 0, 0)))
        assert len(calls) == 1
        assert source.local_stats["requests"] == 0


@pytest.mark.parametrize("records, message", [
    ((2, 0, 8, 1000, 0, 1, 0, 0), "source ID"),
    ((0, 9000, 8, 1000, 0, 1, 0, 0), "exceeds"),
    ((0, 0, 8, 1000, 0, 2, 16, 4), "overlap"),
    ((0, 0, 8, 1000, 0, 1, 0, 0, 1, 0, 8, 1004, 0, 1, 0, 0), "overlapping"),
    ((0, 0, 8), "descriptor size"),
    ((0, 0, 3, 1000, 1, 1, 0, 0), "geometry or expansion"),
    ((0, 0, 8, (1 << 63) - 1, 0, 1, 0, 0), "geometry or expansion"),
])
def test_local_records_keep_admission_checks_before_native(tmp_path, records, message):
    with local_session(tmp_path) as (session, _, calls, _):
        with pytest.raises(ValueError, match=message):
            session._execute(array("Q", records))
        assert calls == []


@pytest.mark.parametrize("failure", ["open", "identity"])
def test_partial_constructor_failure_closes_every_pinned_descriptor(tmp_path, failure):
    with local_session(tmp_path) as (session, source, _, _):
        opened = []
        real_open = os.open

        def open_pinned(path, flags, *args):
            assert str(path).startswith("/proc/self/fd/")
            if failure == "open" and opened:
                raise OSError("injected second descriptor failure")
            if failure == "identity":
                path = f"/proc/self/fd/{source.local.files['b.safetensors'].descriptor}"
            descriptor = real_open(path, flags, *args)
            opened.append(descriptor)
            return descriptor

        with mock.patch("ignition.preparation.remote_weights.os.open", open_pinned), \
                mock.patch("ignition.preparation.remote_weights.tempfile.TemporaryDirectory", side_effect=AssertionError):
            with pytest.raises((OSError, RuntimeError), match="descriptor failure|admitted inode"):
                RemoteWeightSession(source, session.checkpoint, device=0)
        assert opened
        for descriptor in opened:
            with pytest.raises(OSError):
                os.fstat(descriptor)
        for descriptor in session.local_files.values():
            assert os.fstat(descriptor).st_size > 0
