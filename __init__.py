"""ComfyUI custom node: comfyui-mesh.

Distributed FLUX inference across two GPUs on different machines, with
NVENC HEVC compression of activations on the wire between them.

Architecture:
    - This machine (the 5090) runs ComfyUI normally
    - A subset of the FLUX double_blocks runs on a remote 4090 over TCP
    - Activations crossing the wire are compressed via nvenc-pframe

See README.md for setup + the U:/comfyuiserver/ companion script.

Imports use the `from .X import ...` form (package-relative) so that
`codec.py`, `protocol.py`, and `vec_io.py` always resolve to this
folder's copies — not ComfyUI's own top-level `protocol.py` (which
defines an unrelated BinaryEventTypes API).
"""

from .mesh_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
