"""Standalone smoke test for codec.py and protocol.py — does NOT
require ComfyUI to be running, does NOT require a real FLUX model.
Useful for verifying the wire path independently before plugging
the node into a real workflow.

Run from the comfyui-mesh folder:
    python smoke_test_codec.py
"""

from __future__ import annotations

import socket
import threading
import time

import torch

import codec
import protocol


def _make_fake_flux_activation(B=1, T=4096, H=4096, dtype=torch.float16):
    """Make a tensor that looks roughly like a FLUX.2 Klein 9B mid-block
    activation: smooth-ish along the H dimension, structured along T.
    Match the shape we'll see in production."""
    device = torch.device("cuda:0")
    # Mid-block activations are roughly normal with std~1, mean~0; some
    # outlier channels with much wider range. Approximate that.
    base = torch.randn(B, T, H, device=device, dtype=dtype) * 0.3
    # Add a few outlier channels
    outlier_channels = torch.randint(0, H, (16,), device=device)
    base[..., outlier_channels] *= 5.0
    return base


def codec_roundtrip_test():
    print("=== codec round-trip test (no network) ===")
    img = _make_fake_flux_activation()
    print(f"img shape={tuple(img.shape)} dtype={img.dtype} bytes={img.element_size()*img.numel()/1024/1024:.2f} MB")

    for mode_label, kwargs in [
        ("raw", {"mode": "raw"}),
        ("nvenc qp=18", {"mode": "nvenc", "qp": 18, "lossless": False}),
        ("nvenc qp=10", {"mode": "nvenc", "qp": 10, "lossless": False}),
        ("nvenc lossless", {"mode": "nvenc", "qp": 0, "lossless": True}),
    ]:
        try:
            t0 = time.time()
            wire = codec.encode("img", img, **kwargs)
            t_encode = time.time() - t0

            wire_bytes = len(wire.bytes_payload)
            ratio = (img.element_size() * img.numel()) / max(1, wire_bytes)

            t0 = time.time()
            recovered = codec.decode(wire.to_header(), wire.bytes_payload, device=img.device)
            t_decode = time.time() - t0

            if recovered.dtype != img.dtype:
                recovered = recovered.to(img.dtype)
            diff = (recovered.float() - img.float()).abs()
            print(f"  {mode_label:20s}  size={wire_bytes/1024:.1f} KB  ratio={ratio:.2f}x  "
                  f"enc={t_encode*1000:.1f} ms  dec={t_decode*1000:.1f} ms  "
                  f"max_abs_err={float(diff.max()):.4f}")
        except Exception as e:
            print(f"  {mode_label:20s}  FAILED: {type(e).__name__}: {e}")


def loopback_protocol_test(port: int = 17777):
    """Round-trip a small message through a loopback TCP socket to
    verify the protocol layer."""
    print(f"\n=== loopback protocol test (port {port}) ===")

    server_done = threading.Event()
    server_ok = [False]

    def server_thread():
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)
        conn, _addr = s.accept()
        try:
            header, blobs = protocol.recv_message(conn)
            print(f"  server got: kind={header.get('kind')!r}  tensors={len(blobs)}  "
                  f"total_bytes={sum(len(b) for b in blobs)}")
            # Echo back
            protocol.send_message(conn, {"kind": "echo_ack", "tensors": header["tensors"]}, blobs)
            server_ok[0] = True
        finally:
            conn.close()
            s.close()
            server_done.set()

    th = threading.Thread(target=server_thread, daemon=True)
    th.start()
    time.sleep(0.2)

    client = socket.socket()
    client.connect(("127.0.0.1", port))
    img = _make_fake_flux_activation(B=1, T=64, H=64, dtype=torch.float16)
    wire = codec.encode("img", img, mode="nvenc", qp=18)
    protocol.send_message(client, {"kind": "echo_test", "tensors": [wire.to_header()]}, [wire.bytes_payload])

    resp_h, resp_b = protocol.recv_message(client)
    print(f"  client got back: kind={resp_h.get('kind')!r}  tensors={len(resp_b)}  "
          f"total_bytes={sum(len(b) for b in resp_b)}")

    server_done.wait(timeout=5.0)
    client.close()
    if not server_ok[0]:
        print("  loopback FAILED")
    else:
        print("  loopback OK")


if __name__ == "__main__":
    codec_roundtrip_test()
    loopback_protocol_test()
