"""Helpers to flatten and reconstruct the FLUX vec/modulation structure
for the wire protocol.

When `global_modulation=True` (the FLUX.2 path), the `vec` argument
passed to each double_block is a 2-tuple:

    vec = ((img_mod1, img_mod2), (txt_mod1, txt_mod2))

where each `*_mod*` is a `ModulationOut` dataclass with fields
`shift, scale, gate` (each a small tensor of shape [B, 1, hidden_size]).

To ship this over the wire we flatten to a named-tensor list
(12 tensors total). The server reconstructs the structure.

When `global_modulation=False`, `vec` is a single tensor (the encoder
output). We just ship it as one named tensor.
"""

from __future__ import annotations

from typing import Any

import torch


def flatten_vec(vec) -> tuple[str, list[tuple[str, torch.Tensor]]]:
    """Flatten vec into [(name, tensor), ...]. Returns the structure
    descriptor ('tensor' | 'modulation_tuple') and the named tensors."""
    if isinstance(vec, torch.Tensor):
        return "tensor", [("vec", vec)]
    # Otherwise it should be the FLUX.2 nested ModulationOut tuple
    # ((img_mod1, img_mod2), (txt_mod1, txt_mod2))
    img_pair, txt_pair = vec
    img_mod1, img_mod2 = img_pair
    txt_mod1, txt_mod2 = txt_pair
    out = []
    for side_name, mods in (("img", (img_mod1, img_mod2)), ("txt", (txt_mod1, txt_mod2))):
        for mod_idx, mod in enumerate(mods, start=1):
            for field in ("shift", "scale", "gate"):
                t = getattr(mod, field)
                out.append((f"vec_{side_name}_mod{mod_idx}_{field}", t))
    return "modulation_tuple", out


def reconstruct_vec(kind: str, named_tensors: dict[str, torch.Tensor]):
    """Inverse of flatten_vec. For 'modulation_tuple' kind we need to
    materialise ModulationOut instances; we import the canonical dataclass
    from comfy.ldm.flux.layers (works on the server side; the 5090 node
    rarely calls this — it produces the structure, doesn't consume it)."""
    if kind == "tensor":
        return named_tensors["vec"]
    if kind != "modulation_tuple":
        raise ValueError(f"unknown vec kind {kind!r}")
    # Import ModulationOut lazily so this module imports cleanly even
    # without a comfy install (e.g. in protocol-only smoke tests).
    from comfy.ldm.flux.layers import ModulationOut

    def make(side: str, mod_idx: int) -> Any:
        return ModulationOut(
            shift=named_tensors[f"vec_{side}_mod{mod_idx}_shift"],
            scale=named_tensors[f"vec_{side}_mod{mod_idx}_scale"],
            gate=named_tensors[f"vec_{side}_mod{mod_idx}_gate"],
        )

    return (
        (make("img", 1), make("img", 2)),
        (make("txt", 1), make("txt", 2)),
    )


def fake_modulation_vec(B: int, H: int, device, dtype) -> tuple:
    """Construct a fake modulation tuple for smoke-testing without
    running the modulation modules. Shapes match
    `Modulation.forward()` output for global_modulation=True double
    blocks (each ModulationOut field is [B, 1, H])."""
    from comfy.ldm.flux.layers import ModulationOut

    def mk():
        return ModulationOut(
            shift=torch.randn(B, 1, H, device=device, dtype=dtype) * 0.1,
            scale=torch.randn(B, 1, H, device=device, dtype=dtype) * 0.1,
            gate=torch.randn(B, 1, H, device=device, dtype=dtype) * 0.1,
        )

    return ((mk(), mk()), (mk(), mk()))
