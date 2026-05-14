"""Length-prefixed binary protocol for the FLUX mesh wire.

Used by both the ComfyUI custom node (client, runs on the 5090) and
the standalone server script (runs on the 4090).

Wire format — every message:

    [ 4 bytes: total payload length, uint32 big-endian ]
    [ payload ]

Payload format (compact JSON header + concatenated tensor blobs):

    [ 4 bytes: header_len, uint32 big-endian ]
    [ header_len bytes: utf-8 JSON header describing the rest ]
    [ tensor blobs concatenated in the order listed in the header ]

The header looks like:

    {
        "kind": "forward_request" | "forward_response" | "hello" | ...,
        "tensors": [
            {"name": "img", "dtype": "float16", "shape": [...], "encoding": "raw" | "nvenc",
             "size": <bytes>, "extra": {...}},
            ...
        ],
        ... per-kind metadata ...
    }

Keeping the protocol intentionally simple (no msgpack, no pickle) so
the 4090 server can be a single-file script with no extra deps beyond
torch + nvenc-pframe.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any


def _read_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from sock, or raise ConnectionError."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError(f"socket closed with {remaining} bytes still to read")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, header: dict[str, Any], blobs: list[bytes]) -> None:
    """Send one framed message: [total_len][header_len][header_json][blobs...]."""
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    total_len = 4 + len(header_bytes) + sum(len(b) for b in blobs)
    sock.sendall(struct.pack(">I", total_len))
    sock.sendall(struct.pack(">I", len(header_bytes)))
    sock.sendall(header_bytes)
    for b in blobs:
        sock.sendall(b)


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], list[bytes]]:
    """Receive one framed message; return (header, blobs)."""
    total_len = struct.unpack(">I", _read_exact(sock, 4))[0]
    payload = _read_exact(sock, total_len)
    header_len = struct.unpack(">I", payload[:4])[0]
    header = json.loads(payload[4 : 4 + header_len].decode("utf-8"))
    body = payload[4 + header_len:]
    blobs = []
    cursor = 0
    for t in header.get("tensors", []):
        sz = int(t["size"])
        blobs.append(body[cursor : cursor + sz])
        cursor += sz
    if cursor != len(body):
        raise ValueError(f"body had {len(body)} bytes, tensors consumed {cursor}")
    return header, blobs
