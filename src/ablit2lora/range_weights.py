"""Explicit, bounded HTTP ranges for public safetensors files.

This opt-in reader does not replace LazyWeights or download a snapshot. A
server ignoring Range is rejected before its body is read. One reader's byte
budget covers headers and tensor payloads; no hidden retries or fallback.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from urllib.request import Request, urlopen

import torch
from safetensors.torch import load


def hf_tensor_url(repo: str, revision: str, filename: str) -> str:
    """Only immutable public HF revisions, not a mutable main/tag reference."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a pinned 40-character HF commit")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("expected owner/repo")
    if not re.fullmatch(r"[\w./-]+\.safetensors", filename) or ".." in filename:
        raise ValueError("invalid safetensors filename")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


class RangeWeights:
    """One immutable safetensors shard with full-tensor or contiguous-row reads."""

    def __init__(self, url: str, *, max_bytes: int, max_header_bytes: int = 2_097_152):
        if max_bytes < 8 or max_header_bytes < 2:
            raise ValueError("positive header/payload byte budgets required")
        if not url.startswith(("https://", "http://")):
            raise ValueError("only HTTP(S) range sources supported")
        self.url = url
        self.max_bytes = max_bytes
        self.requested_bytes = 0
        self.receipts: list[dict] = []
        self.etag: str | None = None
        self.total_size: int | None = None
        (header_len,) = struct.unpack("<Q", self._read(0, 8))
        if not 2 <= header_len <= max_header_bytes:
            raise ValueError("safetensors header exceeds limit or is invalid")
        self.data_start = 8 + header_len
        self.header = json.loads(self._read(8, header_len))
        if not isinstance(self.header, dict):
            raise ValueError("invalid safetensors header")
        for key, entry in self.header.items():
            if key == "__metadata__":
                continue
            if not isinstance(entry, dict):
                raise ValueError("invalid tensor entry")
            shape = entry.get("shape")
            if (
                not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(entry.get("dtype"), str)
            ):
                raise ValueError("invalid tensor shape/dtype")
            offsets = entry.get("data_offsets", [])
            if (
                not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= self.total_size - self.data_start
            ):
                raise ValueError("tensor offsets outside shard")

    def _read(self, start: int, length: int) -> bytes:
        if start < 0 or length < 1:
            raise ValueError("range must be nonempty and nonnegative")
        if self.requested_bytes + length > self.max_bytes:
            raise ValueError("range byte budget exceeded")
        self.requested_bytes += length
        end = start + length - 1
        req = Request(self.url, headers={"Range": f"bytes={start}-{end}",
                                         "Accept-Encoding": "identity"})
        with urlopen(req, timeout=30) as response:
            if response.status != 206:
                raise ValueError("server ignored bounded Range request (expected 206)")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                 response.headers.get("Content-Range", ""))
            if not match or tuple(map(int, match.groups()[:2])) != (start, end):
                raise ValueError("unexpected Content-Range")
            total = int(match.group(3))
            etag = response.headers.get("ETag")
            if not etag or end >= total:
                raise ValueError("immutable ETag and valid total size required")
            if self.etag is not None and (etag != self.etag or total != self.total_size):
                raise ValueError("remote shard changed between range reads")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("encoded range response not supported")
            payload = response.read(length + 1)
            if len(payload) != length:
                raise ValueError("range response length mismatch")
            self.etag, self.total_size = etag, total
            self.receipts.append({"start": start, "length": length,
                                  "sha256": hashlib.sha256(payload).hexdigest()})
            return payload

    def keys(self) -> list[str]:
        return [name for name in self.header if name != "__metadata__"]

    def get(self, name: str):
        entry = self.header[name]
        lo, hi = entry["data_offsets"]
        return self._tensor(entry, self.data_start + lo, hi - lo)

    def get_rows(self, name: str, start: int, stop: int):
        """Read whole contiguous rows of a byte-aligned 2-D stored tensor.

        No dequantization is implied. Caller reads the corresponding block
        scales separately. Packed sub-byte formats are deliberately refused.
        """
        entry = self.header[name]
        shape = entry.get("shape", [])
        sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8,
                 "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
                 "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}
        itemsize = sizes.get(entry.get("dtype"))
        if len(shape) != 2 or itemsize is None:
            raise ValueError("row reads require a supported byte-aligned 2-D tensor")
        if not 0 <= start < stop <= shape[0]:
            raise ValueError("row bounds outside tensor")
        lo, hi = entry["data_offsets"]
        row_bytes = shape[1] * itemsize
        if hi - lo != shape[0] * row_bytes:
            raise ValueError("stored shape/byte count mismatch")
        sliced = {**entry, "shape": [stop - start, shape[1]]}
        return self._tensor(sliced, self.data_start + lo + start * row_bytes,
                            (stop - start) * row_bytes)

    def _tensor(self, entry: dict, start: int, length: int):
        payload = self._read(start, length)
        if entry["dtype"] == "F8_E8M0":
            # safetensors 0.8's in-memory torch loader lacks this mapping,
            # despite accepting the header. Preserve the typed exponent bits;
            # converting the uint8 values numerically would corrupt scales.
            dtype = getattr(torch, "float8_e8m0fnu", None)
            if dtype is None:
                raise ValueError("typed E8M0 requires a torch version supporting float8_e8m0fnu")
            elements = 1
            for dim in entry["shape"]:
                elements *= dim
            if elements != len(payload):
                raise ValueError("E8M0 shape/byte count mismatch")
            return torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(dtype).reshape(
                entry["shape"])
        meta = {"tensor": {**entry, "data_offsets": [0, length]}}
        header = json.dumps(meta).encode()
        header += b" " * (-len(header) % 8)
        return load(struct.pack("<Q", len(header)) + header + payload)["tensor"]
