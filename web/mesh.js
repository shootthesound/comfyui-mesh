// comfyui-mesh ComfyUI extension.
//
// Renders an inline banner UNDER the MeshSplitFlux node when the
// Python side dispatches a `mesh-message` websocket event (e.g. user
// decreased n_blocks_remote, requires a restart).
//
// The message is stored on the node instance as `_mesh_message_text`.
// The leading underscore keeps litegraph's serializer from saving it
// to the workflow JSON, so messages don't leak across workflow
// reloads (they're per-session and re-emitted by Python anyway).
//
// Drawing happens via a foreground draw hook that paints below the
// node bounds — keeps the node geometry stable so saved workflows
// don't get lopsided heights from a transient banner.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MESH_NODE_NAME = "MeshSplitFlux";

const COLORS = {
    warn:  { fg: "#ffffff", bg: "#a02929" },
    info:  { fg: "#ffffff", bg: "#2c5d99" },
    ok:    { fg: "#ffffff", bg: "#2a7a3a" },
};

function wrapText(ctx, text, maxWidth) {
    const lines = [];
    for (const para of text.split("\n")) {
        if (para === "") { lines.push(""); continue; }
        const words = para.split(/\s+/);
        let line = "";
        for (const word of words) {
            const test = line ? line + " " + word : word;
            if (ctx.measureText(test).width <= maxWidth) {
                line = test;
            } else {
                if (line) lines.push(line);
                line = word;
            }
        }
        if (line) lines.push(line);
    }
    return lines;
}

app.registerExtension({
    name: "comfyui-mesh.NodeBanner",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== MESH_NODE_NAME) return;

        const orig_onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (orig_onDrawForeground) orig_onDrawForeground.apply(this, arguments);
            if (!this._mesh_message_text || this.flags?.collapsed) return;

            const palette = COLORS[this._mesh_message_level] || COLORS.info;
            const padX = 10;
            const padY = 8;
            const lineHeight = 16;
            const fontPx = 12;

            ctx.save();
            ctx.font = `${fontPx}px Segoe UI, sans-serif`;
            const lines = wrapText(ctx, this._mesh_message_text, this.size[0] - 2 * padX);
            const bannerH = padY * 2 + lines.length * lineHeight;
            const bannerY = this.size[1] + 4;

            ctx.fillStyle = palette.bg;
            ctx.beginPath();
            ctx.roundRect(0, bannerY, this.size[0], bannerH, 6);
            ctx.fill();

            ctx.fillStyle = palette.fg;
            ctx.textBaseline = "top";
            for (let i = 0; i < lines.length; i++) {
                ctx.fillText(lines[i], padX, bannerY + padY + i * lineHeight);
            }
            ctx.restore();
        };
    },
});

api.addEventListener("mesh-message", (event) => {
    const { node_id, level, text } = event.detail || {};
    if (!node_id) return;
    const node = app.graph.getNodeById(parseInt(node_id, 10));
    if (!node) return;
    if (level === "clear" || !text) {
        delete node._mesh_message_text;
        delete node._mesh_message_level;
    } else {
        node._mesh_message_text = String(text);
        node._mesh_message_level = level || "info";
    }
    app.graph.setDirtyCanvas(true, true);
});
