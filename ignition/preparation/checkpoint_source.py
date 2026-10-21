"""Serve immutable checkpoint ranges while peer ranks retain only metadata."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import socket
import stat
import struct
import tempfile
import threading
import time


BLOCK_BYTES = 4 << 20
METADATA_BYTES = 256 << 20
MARKER = ".flare-remote-source.json"
_OWNER = ContextVar("flare_checkpoint_source", default=None)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _exact(connection, length):
    data = bytearray()
    while len(data) < length:
        block = connection.recv(min(length - len(data), BLOCK_BYTES))
        if not block:
            raise ConnectionError("checkpoint source disconnected")
        data.extend(block)
    return bytes(data)


def _packet(connection, data):
    if len(data) > BLOCK_BYTES:
        raise ValueError("checkpoint frame exceeds its bounded capacity")
    connection.sendall(struct.pack("<I", len(data)) + data)


def _receive(connection):
    length, = struct.unpack("<I", _exact(connection, 4))
    if length > BLOCK_BYTES:
        raise ValueError("checkpoint frame exceeds its bounded capacity")
    return _exact(connection, length)


def token(path):
    """Read the shared cluster credential without putting it on the wire."""
    path = Path(path).expanduser()
    if path.stat().st_mode & 0o077:
        raise ValueError("checkpoint token file must be private to its owner")
    value = path.read_bytes().strip()
    if len(value) < 32:
        raise ValueError("checkpoint token must contain at least 32 bytes")
    return value


class Channel:
    """Authenticate each bounded frame, its direction, and its sequence number."""

    def __init__(self, connection, key, *, server):
        self.connection, self.key = connection, key
        self.send_role, self.receive_role = (b"root", b"peer") if server else (b"peer", b"root")
        self.sent = self.received = 0

    def send(self, kind, payload=b"", **fields):
        header = _json(dict(kind=kind, sequence=self.sent, bytes=len(payload), **fields))
        if len(header) > 65536 or len(payload) > BLOCK_BYTES:
            raise ValueError("checkpoint message exceeds its bounded capacity")
        signature = hmac.digest(self.key, self.send_role + header + payload, "sha256")
        _packet(self.connection, header)
        self.connection.sendall(payload + signature)
        self.sent += 1

    def receive(self, expected=None):
        header = _receive(self.connection)
        if len(header) > 65536:
            raise ValueError("checkpoint message header exceeds capacity")
        message = json.loads(header)
        length = message.get("bytes")
        if type(length) is not int or not 0 <= length <= BLOCK_BYTES:
            raise ValueError("invalid checkpoint payload length")
        payload, signature = _exact(self.connection, length), _exact(self.connection, 32)
        actual = hmac.digest(self.key, self.receive_role + header + payload, "sha256")
        if not hmac.compare_digest(signature, actual):
            raise ValueError("checkpoint message authentication failed")
        if message.get("sequence") != self.received:
            raise ValueError("checkpoint message sequence differs")
        if expected is not None and message.get("kind") != expected:
            raise ValueError(f"expected checkpoint {expected}, received {message.get('kind')}")
        self.received += 1
        if message.get("kind") == "error":
            raise RuntimeError(f"checkpoint source rejected access: {message.get('error')}")
        return message, payload


def _proof(key, role, challenge, nonce):
    return hmac.digest(key, role + _json(challenge) + nonce.encode(), "sha256").hex()


def authenticate(connection, key, *, rank, identity, catalog=None):
    """Bind both peers to fresh nonces, rank, source code and catalog identity."""
    server = catalog is not None
    if server:
        challenge = dict(version=1, rank=rank, identity=identity, nonce=secrets.token_hex(32),
                         catalog_sha256=hashlib.sha256(_json(catalog)).hexdigest())
        _packet(connection, _json(challenge))
        reply = json.loads(_receive(connection))
        nonce = reply.get("nonce", "")
        if len(nonce) != 64 or not hmac.compare_digest(reply.get("proof", ""), _proof(key, b"peer", challenge, nonce)):
            raise ValueError("checkpoint peer authentication failed")
        _packet(connection, _json(dict(proof=_proof(key, b"root", challenge, nonce))))
    else:
        challenge = json.loads(_receive(connection))
        if (challenge.get("version"), challenge.get("rank"), challenge.get("identity")) != (1, rank, identity):
            raise ValueError("checkpoint source rank or code identity differs")
        nonce = secrets.token_hex(32)
        _packet(connection, _json(dict(nonce=nonce, proof=_proof(key, b"peer", challenge, nonce))))
        reply = json.loads(_receive(connection))
        if not hmac.compare_digest(reply.get("proof", ""), _proof(key, b"root", challenge, nonce)):
            raise ValueError("checkpoint root authentication failed")
    session = hmac.digest(key, b"checkpoint-session" + _json(challenge) + nonce.encode(), "sha256")
    return Channel(connection, session, server=server), challenge["catalog_sha256"]


def relative_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
        raise ValueError("checkpoint path must be a canonical source-relative path")
    return name


def file_identity(descriptor):
    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("checkpoint source must be a regular file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def row_extent(size, offset, width, rows, stride):
    if any(type(value) is not int or value < 0 or value >= 1 << 63 for value in (size, offset, width, rows, stride)):
        raise ValueError("checkpoint row geometry must use nonnegative signed integers")
    if rows < 1 or width * rows > BLOCK_BYTES:
        raise ValueError("checkpoint request exceeds its bounded capacity")
    extent = (rows - 1) * stride + width
    if extent > BLOCK_BYTES or offset + extent > size:
        raise ValueError("checkpoint request lies outside its source or bounded extent")
    return extent


@dataclass
class HeldFile:
    path: Path
    descriptor: int
    identity: tuple

    def validate(self):
        if file_identity(self.descriptor) != self.identity or file_identity_path(self.path) != self.identity:
            raise RuntimeError("checkpoint source changed after admission")


def file_identity_path(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def release_read_pages(descriptor, offset=0, length=0):
    """Release completed buffered reads from the integrated GPU's shared RAM budget."""
    page = os.sysconf("SC_PAGESIZE")
    first = offset // page * page
    size = ((offset + length + page - 1) // page * page - first) if length else 0
    os.posix_fadvise(descriptor, first, size, os.POSIX_FADV_DONTNEED)


class RootSource:
    """Own authenticated source descriptors until all peer loading sessions end."""

    def __init__(self, snapshot, checkpoint):
        self.snapshot = Path(snapshot).resolve()
        if (self.snapshot / MARKER).exists():
            raise ValueError("a metadata-only remote source cannot publish a checkpoint")
        self.files = {}
        entries, metadata_bytes = [], 0
        try:
            for path in sorted(self.snapshot.rglob("*")):
                if not path.is_file():
                    continue
                name = relative_name(path.relative_to(self.snapshot).as_posix())
                descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                held = HeldFile(path, descriptor, file_identity(descriptor))
                self.files[name] = held
                # Selective reads release their completed ranges; read-ahead outside those ranges
                # would remain cached in the same DRAM needed by the integrated GPU.
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_RANDOM)
                size = held.identity[2]
                header = None
                if name.endswith(".safetensors"):
                    prefix = os.pread(descriptor, 8, 0)
                    header = 8 + int.from_bytes(prefix, "little")
                    if len(prefix) != 8 or not 8 < header <= min(size, (100 << 20) + 8):
                        raise ValueError(f"invalid safetensors source header: {name}")
                extent = size if header is None else header
                metadata_bytes += extent
                if metadata_bytes > METADATA_BYTES:
                    raise ValueError("checkpoint metadata and headers exceed the bounded admission budget")
                digest = hashlib.sha256()
                for offset in range(0, extent, BLOCK_BYTES):
                    data = os.pread(descriptor, min(BLOCK_BYTES, extent - offset), offset)
                    if len(data) != min(BLOCK_BYTES, extent - offset):
                        raise ValueError("checkpoint metadata read was truncated")
                    digest.update(data)
                held.validate()
                release_read_pages(descriptor)
                entries.append(dict(path=name, bytes=size, header_bytes=header, sha256=digest.hexdigest()))
            self.catalog = dict(repository=checkpoint.repository, revision=checkpoint.revision,
                                name=self.snapshot.name, files=entries)
            if len(_json(self.catalog)) > BLOCK_BYTES:
                raise ValueError("checkpoint catalog exceeds capacity")
        except BaseException:
            self.close()
            raise

    def digest(self, name):
        """Hash an admitted file with bounded memory and immutable descriptor identity."""
        held = self.files[relative_name(name)]
        held.validate()
        digest = hashlib.sha256()
        for offset in range(0, held.identity[2], BLOCK_BYTES):
            size = min(BLOCK_BYTES, held.identity[2] - offset)
            data = os.pread(held.descriptor, size, offset)
            if len(data) != size:
                raise ConnectionError("checkpoint digest read was truncated")
            digest.update(data)
            release_read_pages(held.descriptor, offset, size)
        held.validate()
        return digest.hexdigest()

    def read(self, name, offset, width, rows=1, stride=0):
        held = self.files[relative_name(name)]
        extent = row_extent(held.identity[2], offset, width, rows, stride)
        held.validate()
        raw = os.pread(held.descriptor, extent, offset)
        if len(raw) != extent:
            raise ConnectionError("checkpoint source read was truncated")
        held.validate()
        result = b"".join(raw[index * stride:index * stride + width] for index in range(rows))
        release_read_pages(held.descriptor, offset, extent)
        return result

    def close(self):
        for held in self.files.values():
            release_read_pages(held.descriptor)
            os.close(held.descriptor)
        self.files.clear()


class SourceServer:
    """Sole reader of a peer control socket until its authenticated loading drain."""

    def __init__(self, source, connection, key, rank, identity):
        self.source, self.connection, self.key, self.rank, self.identity = source, connection, key, rank, identity
        self.ready = threading.Event()
        self.error = None
        self.stats = dict(payload_bytes=0, metadata_bytes=0, requests=0)
        self.thread = threading.Thread(target=self.run, name=f"flare-checkpoint-rank{rank}", daemon=True)

    def run(self):
        try:
            channel, _ = authenticate(self.connection, self.key, rank=self.rank, identity=self.identity,
                                      catalog=self.source.catalog)
            channel.send("catalog", _json(self.source.catalog))
            while True:
                message, payload = channel.receive()
                if payload:
                    raise ValueError("checkpoint requests cannot carry payload bytes")
                kind = message["kind"]
                if kind == "ready":
                    channel.send("ready")
                    self.ready.set()
                elif kind == "finished":
                    if not self.ready.is_set() or message["stats"] != self.stats:
                        raise ValueError("checkpoint source completion byte accounting differs")
                    for held in self.source.files.values():
                        held.validate()
                    channel.send("finished", stats=self.stats)
                    return
                elif kind == "sha256":
                    name = relative_name(message["name"])
                    channel.send("sha256", name=name, sha256=self.source.digest(name))
                elif kind == "read":
                    fields = message["read"]
                    data = self.source.read(**fields)
                    self.stats["payload_bytes" if fields["name"].endswith(".safetensors") and
                               fields["offset"] >= next(e["header_bytes"] for e in self.source.catalog["files"]
                                                       if e["path"] == fields["name"]) else "metadata_bytes"] += len(data)
                    self.stats["requests"] += 1
                    channel.send("data", data)
                else:
                    raise ValueError(f"unknown checkpoint source operation {kind}")
        except BaseException as error:
            self.error = error
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        finally:
            self.ready.set()

    def wait_ready(self):
        self.ready.wait()
        if self.error is not None:
            raise RuntimeError(f"rank {self.rank} checkpoint admission failed") from self.error

    def join(self):
        self.thread.join()
        if self.error is not None:
            raise RuntimeError(f"rank {self.rank} checkpoint reader failed") from self.error


class RemoteSource:
    """Authenticate remote metadata, then use requested ranges or an exact local mirror."""

    def __init__(self, connection, key, rank, identity, checkpoint, *, directory=None, on_created=None,
                 local_snapshot=None, progress=None):
        self.local = None
        self.channel, digest = authenticate(connection, key, rank=rank, identity=identity)
        _, payload = self.channel.receive("catalog")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("checkpoint catalog identity differs")
        catalog = json.loads(payload)
        if (catalog["repository"], catalog["revision"]) != (checkpoint.repository, checkpoint.revision):
            raise ValueError("checkpoint source differs from its preparation assignment")
        self.files = {relative_name(entry["path"]): entry for entry in catalog["files"]}
        if len(self.files) != len(catalog["files"]):
            raise ValueError("duplicate checkpoint source path")
        metadata = 0
        for entry in self.files.values():
            size, header = entry["bytes"], entry["header_bytes"]
            if type(size) is not int or not 0 <= size < 1 << 63:
                raise ValueError("invalid checkpoint source extent")
            if header is not None and (type(header) is not int or not 8 < header <= min(size, (100 << 20) + 8)):
                raise ValueError("invalid checkpoint header extent")
            metadata += size if header is None else header
        if metadata > METADATA_BYTES:
            raise ValueError("checkpoint metadata exceeds admission capacity")
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="flare-checkpoint-", dir=directory)
        if on_created is not None:
            on_created(self)
        name = relative_name(catalog["name"])
        if "/" in name:
            raise ValueError("checkpoint snapshot name must be a basename")
        self.snapshot = Path(self.temporary.name) / name
        self.snapshot.mkdir()
        self.headers = {}
        self.stats = dict(payload_bytes=0, metadata_bytes=0, requests=0)
        self.local_stats = dict(payload_bytes=0, metadata_bytes=0, requests=0)
        self.finished = False
        (self.snapshot / MARKER).write_text(json.dumps(dict(catalog_sha256=digest)))
        try:
            for name, entry in self.files.items():
                size = entry["bytes"] if entry["header_bytes"] is None else entry["header_bytes"]
                content = bytearray()
                for offset in range(0, size, BLOCK_BYTES):
                    content.extend(self.read(name, offset, min(BLOCK_BYTES, size - offset)))
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise ValueError("checkpoint metadata fingerprint differs")
                if entry["header_bytes"] is not None:
                    self.headers[name] = bytes(content)
                else:
                    target = self.snapshot / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
            if local_snapshot is not None:
                self.local = RootSource(local_snapshot, checkpoint)
                if self.local.catalog != catalog:
                    raise ValueError("local checkpoint catalog differs from the authenticated source")
                total_bytes = sum(entry["bytes"] for entry in self.files.values())
                validated_bytes, reported = 0, time.monotonic()
                if progress is not None:
                    progress(f"verify local checkpoint: 0/{len(self.files)} files, 0.00/{total_bytes / (1 << 30):.2f} GiB")
                for index, name in enumerate(self.files, 1):
                    # The coordinator hashes concurrently with the peer; the response is HMAC bound.
                    self.channel.send("sha256", name=name)
                    local_digest = self.local.digest(name)
                    message, payload = self.channel.receive("sha256")
                    if payload or message.get("name") != name or message.get("sha256") != local_digest:
                        raise ValueError(f"local checkpoint whole-file SHA256 differs: {name}")
                    validated_bytes += self.files[name]["bytes"]
                    now = time.monotonic()
                    if progress is not None and (now - reported >= 5 or index == len(self.files)):
                        progress(f"verified local checkpoint: {index}/{len(self.files)} files, "
                                 f"{validated_bytes / (1 << 30):.2f}/{total_bytes / (1 << 30):.2f} GiB")
                        reported = now
            self.channel.send("ready")
            self.channel.receive("ready")
        except BaseException:
            self.close()
            raise

    @contextmanager
    def activate(self):
        if self.local is not None:
            for held in self.local.files.values():
                held.validate()
        value = _OWNER.set(self)
        try:
            yield self.snapshot
        finally:
            _OWNER.reset(value)

    def name(self, path):
        return Path(path).resolve().relative_to(self.snapshot).as_posix()

    def read(self, name, offset, width, rows=1, stride=0):
        if self.finished:
            raise RuntimeError("checkpoint reader has already drained")
        entry = self.files[relative_name(name)]
        row_extent(entry["bytes"], offset, width, rows, stride)
        if self.local is not None:
            data = self.local.read(name, offset, width, rows, stride)
            stats = self.local_stats
        else:
            self.channel.send("read", read=dict(name=name, offset=offset, width=width, rows=rows, stride=stride))
            _, data = self.channel.receive("data")
            stats = self.stats
        if len(data) != width * rows:
            raise ConnectionError("checkpoint source returned a different byte extent")
        stats["payload_bytes" if entry["header_bytes"] is not None and offset >= entry["header_bytes"]
              else "metadata_bytes"] += len(data)
        stats["requests"] += 1
        return data

    def finish(self):
        if self.local is not None:
            for held in self.local.files.values():
                held.validate()
        self.channel.send("finished", stats=self.stats)
        message, _ = self.channel.receive("finished")
        if message["stats"] != self.stats:
            raise ValueError("checkpoint completion accounting differs")
        self.finished = True

    def close(self):
        if self.local is not None:
            self.local.close()
            self.local = None
        self.temporary.cleanup()


def source_for(path):
    """Resolve only the active owner's paths; marked directories require that owner."""
    path = Path(path).resolve()
    owner = _OWNER.get()
    if owner is not None and path.is_relative_to(owner.snapshot):
        return owner
    for parent in (path, *path.parents):
        if (parent / MARKER).is_file():
            raise RuntimeError("remote checkpoint access requires its active source owner")
    return None


def source_size(path):
    owner = source_for(path)
    return Path(path).stat().st_size if owner is None else owner.files[owner.name(path)]["bytes"]


def tensor_dtype(name):
    import torch

    names = {"BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "I32": "int32",
             "I64": "int64", "U16": "uint16", "U32": "uint32", "U64": "uint64", "F16": "float16",
             "BF16": "bfloat16", "F32": "float32", "F64": "float64", "F8_E4M3": "float8_e4m3fn",
             "F8_E5M2": "float8_e5m2", "F8_E8M0": "float8_e8m0fnu"}
    return getattr(torch, names[name])


def read_small_tensor(manifest, name, *, limit):
    """Read a declared remote CPU scalar without opening a nonexistent tensor file."""
    import torch

    source = source_for(manifest.snapshot)
    if source is None:
        raise ValueError("remote scalar reads require an owned source")
    tensor = manifest.tensors[name]
    if tensor.nbytes > min(limit, BLOCK_BYTES):
        raise ValueError("remote scalar exceeds its declared read budget")
    if not tensor.nbytes:
        return torch.empty(tensor.shape, dtype=tensor_dtype(tensor.dtype), device="cpu")
    content = bytearray(source.read(tensor.shard, tensor.offset, tensor.nbytes))
    return torch.frombuffer(content, dtype=tensor_dtype(tensor.dtype)).reshape(tensor.shape)
