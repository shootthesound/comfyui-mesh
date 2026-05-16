"""ComfyUI custom node: comfyui-mesh.

Distributed FLUX inference across two GPUs on different machines, with
NVENC HEVC compression of activations on the wire between them.

Architecture:
    - This machine (the 5090) runs ComfyUI normally
    - A subset of the FLUX double_blocks runs on a remote 4090 over TCP
    - Activations crossing the wire are compressed via nvenc-pframe
      (bundled in `./nvenc_pframe/` — no separate install required)

See README.md for setup + the server/ folder for the back-half deploy.

Imports use the `from .X import ...` form (package-relative) so that
`codec.py`, `protocol.py`, and `vec_io.py` always resolve to this
folder's copies — not ComfyUI's own top-level `protocol.py` (which
defines an unrelated BinaryEventTypes API).
"""

# Put this folder on sys.path so the bundled `nvenc_pframe/` subpackage
# can be imported as a top-level `import nvenc_pframe` from codec.py.
# codec.py is mirrored byte-identical to the server side, so it uses
# the absolute `from nvenc_pframe.direct.backend import DirectBackend`
# form there too (server's mesh_server.py does the same sys.path setup).
import sys
from pathlib import Path
_HERE = str(Path(__file__).parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from .mesh_node import (  # noqa: E402
    NODE_CLASS_MAPPINGS as _FLUX_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _FLUX_NODE_DISPLAY_NAME_MAPPINGS,
)
from .mesh_node_ltx import (  # noqa: E402
    NODE_CLASS_MAPPINGS as _LTX_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _LTX_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {**_FLUX_NODE_CLASS_MAPPINGS, **_LTX_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_FLUX_NODE_DISPLAY_NAME_MAPPINGS,
    **_LTX_NODE_DISPLAY_NAME_MAPPINGS,
}

# Tells ComfyUI to serve the contents of ./web/ alongside the nodes.
# `web/mesh.js` handles the FLUX (Icarus) node UI; `web/mesh_ltx.js`
# handles the LTX (Icarus LTX) node UI. Both listen for `mesh-message`
# websocket events and render inline banners.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
