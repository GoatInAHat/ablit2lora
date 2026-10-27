import io
import json
import struct
from email.message import Message

import pytest
import torch
from safetensors.torch import save

from ablit2lora.range_weights import RangeWeights, hf_tensor_url


class Response(io.BytesIO):
    def __init__(self, payload, status, headers):
        super().__init__(payload)
        self.status = status
        self.headers = Message()
        for key, value in headers.items():
            self.headers[key] = value
        self.read_count = 0

    def read(self, size=-1):
        self.read_count += 1
        return super().read(size)


def source(monkeypatch, *, ignore=False, changed=False, bad_range=False):
    matrix = torch.arange(120, dtype=torch.float32).reshape(10, 12)
    data = save({"writer.weight": matrix, "other": torch.ones(1000)})
    responses = []

    def open_range(req, timeout):
        lo, hi = map(int, req.headers["Range"].removeprefix("bytes=").split("-"))
        payload = data[lo:hi + 1]
        headers = {"Content-Range": f"bytes {lo}-{hi}/{len(data)}", "ETag": '"stable"'}
        if bad_range:
            headers["Content-Range"] = "bytes 1-8/10000"
        if changed and responses:
            headers["ETag"] = '"new"'
        response = Response(data if ignore else payload, 200 if ignore else 206, headers)
        responses.append(response)
        return response

    monkeypatch.setattr("ablit2lora.range_weights.urlopen", open_range)
    return matrix, data, responses


def test_rows_and_tensor_are_byte_bounded_and_match_local_decode(monkeypatch):
    matrix, data, responses = source(monkeypatch)
    reader = RangeWeights("https://example.test/a.safetensors", max_bytes=2048)
    assert torch.equal(reader.get_rows("writer.weight", 3, 5), matrix[3:5])
    assert torch.equal(reader.get("writer.weight"), matrix)
    assert reader.requested_bytes < len(data)
    assert len(responses) == 4
    assert sum(r["length"] for r in reader.receipts) == reader.requested_bytes
    assert all(len(r["sha256"]) == 64 for r in reader.receipts)
    with pytest.raises(ValueError, match="budget"):
        reader.get("other")
    assert len(responses) == 4  # oversized request refused before network


def test_server_ignoring_range_is_rejected_without_reading_body(monkeypatch):
    _, _, responses = source(monkeypatch, ignore=True)
    with pytest.raises(ValueError, match="ignored"):
        RangeWeights("https://example.test/a.safetensors", max_bytes=2048)
    assert responses[0].read_count == 0


@pytest.mark.parametrize("options", [{"changed": True}, {"bad_range": True}])
def test_changing_identity_or_incorrect_range_is_refused(monkeypatch, options):
    source(monkeypatch, **options)
    with pytest.raises(ValueError):
        RangeWeights("https://example.test/a.safetensors", max_bytes=2048)


def test_invalid_header_cannot_trigger_unbounded_read(monkeypatch):
    calls = []

    def open_range(req, timeout):
        calls.append(req)
        return Response(struct.pack("<Q", 1 << 40), 206,
                        {"Content-Range": "bytes 0-7/1099511627784", "ETag": '"stable"'})

    monkeypatch.setattr("ablit2lora.range_weights.urlopen", open_range)
    with pytest.raises(ValueError, match="header"):
        RangeWeights("https://example.test/a.safetensors", max_bytes=2048)
    assert len(calls) == 1


def test_immutable_hf_url_and_row_shape_guards(monkeypatch):
    url = hf_tensor_url("owner/repo", "a" * 40, "model-1.safetensors")
    assert "/resolve/" + "a" * 40 + "/" in url
    with pytest.raises(ValueError, match="pinned"):
        hf_tensor_url("owner/repo", "main", "model.safetensors")
    source(monkeypatch)
    reader = RangeWeights(url, max_bytes=2048)
    with pytest.raises(ValueError, match="row bounds"):
        reader.get_rows("writer.weight", -1, 1)
    with pytest.raises(ValueError, match="2-D"):
        reader.get_rows("other", 0, 1)
    assert json.dumps(reader.receipts)


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="torch lacks typed E8M0")
def test_real_e8m0_scale_bytes_are_typed_not_integer_values(monkeypatch):
    expected = torch.tensor([[0.5, 1., 2., 4.], [8., 16., 32., 64.]])
    data = save({"scale": expected.to(torch.float8_e8m0fnu)})

    def open_range(req, timeout):
        lo, hi = map(int, req.headers["Range"].removeprefix("bytes=").split("-"))
        return Response(data[lo:hi + 1], 206,
                        {"Content-Range": f"bytes {lo}-{hi}/{len(data)}", "ETag": '"stable"'})

    monkeypatch.setattr("ablit2lora.range_weights.urlopen", open_range)
    reader = RangeWeights("https://example.test/scale.safetensors", max_bytes=2048)
    assert torch.equal(reader.get_rows("scale", 1, 2).float(), expected[1:2])
    assert torch.equal(reader.get("scale").float(), expected)
