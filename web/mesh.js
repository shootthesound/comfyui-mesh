// comfyui-mesh — ComfyUI UI extension.
//
// Two responsibilities:
//
// 1. Inline message banner under the node when Python sends a
//    `mesh-message` websocket event (e.g. user decreased
//    n_blocks_remote — must restart ComfyUI). Banner state lives on
//    `node._mesh_message_text` (leading underscore = litegraph won't
//    serialize, so banner doesn't leak across workflow saves/loads).
//
// 2. Pill-style canvas-drawn widgets in place of the framework Vue
//    widgets. Three reasons we go custom-typed:
//      - Vue number/button widgets bottom-align text on some setups;
//        canvas-drawn pills centre cleanly.
//      - Consistent visual language across the node so future controls
//        slot in without looking foreign.
//      - Picker modal for combos > scrolling the framework dropdown
//        with a tiny chevron.
//
// Serialization-stability rules followed throughout (so workflows save
// + load without column drift, regardless of how many widgets we
// add/remove in the future):
//
//   - ALWAYS replace a framework widget at its EXACT original index
//     in node.widgets. widgets_values is positional; if our pill
//     widget lands in a different slot, every other widget's saved
//     value lands in the wrong widget on load.
//   - NEVER conditionally splice widgets out. If a control should hide
//     itself, return `[0, -4]` from computeSize (zero-height = invisible)
//     instead of removing from the array.
//   - KEEP the widget `name` matching the Python INPUT_TYPES key —
//     ComfyUI submits widget values to the backend by name, not index.
//   - DON'T set `serialize=false` on pill widgets. Letting them serialize
//     naturally keeps `widgets_values.length === widgets.length`.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MESH_NODE_NAME = "MeshSplitFlux";

// Slot height for our canvas-drawn widgets. Must match the height we
// return from computeSize() — LiteGraph's `wh` argument to draw() is a
// fixed constant (~20), not the slot height we asked for.
const SLOT_H = 30;

const PILL_COLORS = {
    body:    "#1f1f1f",
    border:  "#555",
    label:   "#bbb",
    value:   "#fff",
    valueDim: "#888",
    arrow:   "#aaa",
    boolOn:  "#2a7a3a",
    boolOff: "#444",
};

const BANNER_COLORS = {
    warn: { fg: "#ffffff", bg: "#a02929" },
    info: { fg: "#ffffff", bg: "#2c5d99" },
    ok:   { fg: "#ffffff", bg: "#2a7a3a" },
};

// =====================================================================
// Drawing primitives
// =====================================================================

function drawPillBackground(ctx, x, y, w, h) {
    ctx.fillStyle = PILL_COLORS.body;
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = PILL_COLORS.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
}

function ellipsize(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) return text;
    let cut = text;
    while (cut.length > 1 && ctx.measureText(cut + "…").width > maxWidth) {
        cut = cut.slice(0, -1);
    }
    return cut + "…";
}

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

// =====================================================================
// Picker modal — used by pill combos
//
// Plain-DOM modal with a search input + scrollable result list.
// Empty query → alphabetical-ish list with current selection
// highlighted; type to filter (substring match — combos are short, no
// fzf needed). ↑/↓ + Enter, Esc cancels.
// =====================================================================

function showPickerModal(items, current, opts = {}) {
    const { title = "Pick", onSelect, placeholder = "Type to filter…" } = opts;

    const backdrop = document.createElement("div");
    backdrop.style.cssText = `
        position: fixed; inset: 0; background: rgba(0,0,0,0.55);
        display: flex; align-items: center; justify-content: center;
        z-index: 10000; font-family: Segoe UI, sans-serif;
    `;

    const modal = document.createElement("div");
    modal.style.cssText = `
        background: #2a2a2a; color: #ddd; border: 1px solid #555;
        border-radius: 8px; padding: 14px; width: 460px; max-width: 90vw;
        max-height: 70vh; display: flex; flex-direction: column; gap: 8px;
    `;

    const header = document.createElement("div");
    header.style.cssText = "font-size: 13px; font-weight: 600; color: #ccc;";
    header.textContent = title;

    const input = document.createElement("input");
    input.type = "text";
    input.placeholder = placeholder;
    input.style.cssText = `
        background: #1a1a1a; color: #eee; border: 1px solid #555;
        padding: 6px 10px; font-size: 13px; border-radius: 4px;
    `;

    const results = document.createElement("div");
    results.style.cssText = `
        flex: 1; overflow-y: auto; background: #1a1a1a; border: 1px solid #444;
        border-radius: 4px; min-height: 120px; max-height: 40vh;
        font-size: 12px; font-family: monospace;
    `;

    modal.appendChild(header);
    modal.appendChild(input);
    modal.appendChild(results);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);

    let selectedIdx = 0;
    let currentResults = [];

    const close = () => backdrop.remove();
    const choose = (val) => {
        close();
        if (val !== null && val !== undefined && onSelect) onSelect(val);
    };

    const updateHighlight = () => {
        Array.from(results.children).forEach((row, i) => {
            row.style.background = i === selectedIdx ? "#3a3a3a" : "transparent";
        });
        const sel = results.children[selectedIdx];
        if (sel) sel.scrollIntoView({ block: "nearest" });
    };

    const render = () => {
        const q = input.value.trim().toLowerCase();
        currentResults = (q
            ? items.filter((it) => String(it).toLowerCase().includes(q))
            : items.slice()
        );
        const curIdx = currentResults.findIndex((it) => String(it) === String(current));
        selectedIdx = curIdx >= 0 ? curIdx : 0;
        results.innerHTML = "";
        currentResults.forEach((it, i) => {
            const row = document.createElement("div");
            const isCurrent = String(it) === String(current);
            row.style.cssText = `
                padding: 5px 10px; cursor: pointer; user-select: none;
                ${isCurrent ? "color: #4af;" : ""}
            `;
            row.textContent = String(it);
            row.addEventListener("click", () => choose(it));
            row.addEventListener("mouseenter", () => {
                selectedIdx = i;
                updateHighlight();
            });
            results.appendChild(row);
        });
        if (currentResults.length === 0) {
            const empty = document.createElement("div");
            empty.style.cssText = "padding: 16px; text-align: center; color: #888;";
            empty.textContent = "no matches";
            results.appendChild(empty);
        }
        updateHighlight();
    };

    input.addEventListener("input", render);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Escape") { e.preventDefault(); close(); }
        else if (e.key === "ArrowDown") {
            e.preventDefault();
            selectedIdx = Math.min(selectedIdx + 1, currentResults.length - 1);
            updateHighlight();
        }
        else if (e.key === "ArrowUp") {
            e.preventDefault();
            selectedIdx = Math.max(selectedIdx - 1, 0);
            updateHighlight();
        }
        else if (e.key === "Enter") {
            e.preventDefault();
            const r = currentResults[selectedIdx];
            if (r !== undefined) choose(r);
        }
    });

    backdrop.addEventListener("click", (e) => {
        if (e.target === backdrop) close();
    });

    render();
    setTimeout(() => input.focus(), 0);
}

// =====================================================================
// Pill widget builders
//
// Each builder REPLACES a framework widget at the same node.widgets
// index, preserving the original `name` (so backend submission still
// works) and seeding `value` from the framework widget's current value
// (so saved workflows load correctly).
// =====================================================================

function _replaceFrameworkWidget(node, name, builder) {
    const fw = node.widgets?.find((w) => w.name === name);
    if (!fw) return null;
    const fwIdx = node.widgets.indexOf(fw);
    const widget = builder(fw);
    if (!widget) return null;
    if (fwIdx >= 0) node.widgets.splice(fwIdx, 1);
    const insertAt = fwIdx >= 0 ? fwIdx : node.widgets.length;
    node.widgets.splice(insertAt, 0, widget);
    return widget;
}

function createPillNumber(node, name, opts = {}) {
    return _replaceFrameworkWidget(node, name, (fw) => {
        const fwOpts = fw.options || {};
        const min = opts.min ?? fwOpts.min ?? -Infinity;
        const max = opts.max ?? fwOpts.max ?? Infinity;
        const step = opts.step ?? fwOpts.step ?? 1;
        const isInt = opts.integer ?? Number.isInteger(step);
        const label = opts.label ?? name;
        const initialValue = (typeof fw.value === "number")
            ? fw.value
            : (isInt ? 0 : 0.0);
        const origCallback = fw.callback;

        return {
            type: "MESH_PILL_NUMBER",
            name: name,
            label: label,
            value: initialValue,
            options: { ...fwOpts, min, max, step, integer: isInt },
            _mesh_role: `num_${name}`,
            _mesh_drag: null,

            draw(ctx, n, ww, y) {
                const h = SLOT_H;
                const padX = 8;
                const arrowW = 16;
                drawPillBackground(ctx, padX, y + 2, ww - padX * 2, h - 4);

                ctx.textBaseline = "middle";
                ctx.font = "12px Segoe UI, Arial";

                ctx.fillStyle = PILL_COLORS.arrow;
                ctx.textAlign = "center";
                ctx.fillText("◀", padX + arrowW / 2 + 4, y + h / 2);
                ctx.fillText("▶", ww - padX - arrowW / 2 - 4, y + h / 2);

                ctx.fillStyle = PILL_COLORS.label;
                ctx.textAlign = "left";
                const labelX = padX + arrowW + 12;
                const labelText = ellipsize(ctx, this.label, ww * 0.5);
                ctx.fillText(labelText, labelX, y + h / 2);

                ctx.fillStyle = PILL_COLORS.value;
                ctx.textAlign = "right";
                const valStr = this.options.integer
                    ? String(Math.round(this.value))
                    : this.value.toFixed(2);
                ctx.fillText(valStr, ww - padX - arrowW - 12, y + h / 2);
            },

            computeSize() { return [0, SLOT_H]; },

            _clamp(v) {
                return Math.max(this.options.min, Math.min(this.options.max, v));
            },

            _setValue(v) {
                const clamped = this._clamp(v);
                const stepped = Math.round(clamped / this.options.step) * this.options.step;
                this.value = this.options.integer
                    ? Math.round(stepped)
                    : parseFloat(stepped.toFixed(4));
                if (origCallback) {
                    try { origCallback.call(this, this.value); } catch (e) { /* noop */ }
                }
            },

            mouse(event, pos, n) {
                const x = pos[0];
                const ww = n.size[0];
                const padX = 8;
                const arrowHit = 24;
                const onLeft = x < padX + arrowHit;
                const onRight = x > ww - padX - arrowHit;

                if (event.type === "pointerdown" || event.type === "mousedown") {
                    if (onLeft) {
                        this._setValue(this.value - this.options.step);
                        n.setDirtyCanvas(true, true);
                        return true;
                    }
                    if (onRight) {
                        this._setValue(this.value + this.options.step);
                        n.setDirtyCanvas(true, true);
                        return true;
                    }
                    if (event.detail === 2) {
                        const entered = window.prompt(`${this.label}:`, String(this.value));
                        if (entered !== null) {
                            const parsed = this.options.integer
                                ? parseInt(entered, 10)
                                : parseFloat(entered);
                            if (!isNaN(parsed)) {
                                this._setValue(parsed);
                                n.setDirtyCanvas(true, true);
                            }
                        }
                        this._mesh_drag = null;
                        return true;
                    }
                    this._mesh_drag = { startX: x, startValue: this.value };
                    return true;
                }
                if (event.type === "pointermove" || event.type === "mousemove") {
                    if (this._mesh_drag) {
                        const dx = x - this._mesh_drag.startX;
                        // 1 step per 4 px of drag.
                        const delta = dx * this.options.step * 0.25;
                        this._setValue(this._mesh_drag.startValue + delta);
                        n.setDirtyCanvas(true, true);
                        return true;
                    }
                }
                if (event.type === "pointerup" || event.type === "mouseup") {
                    if (this._mesh_drag) {
                        this._mesh_drag = null;
                        return true;
                    }
                }
                return false;
            },
        };
    });
}

function createPillCombo(node, name, opts = {}) {
    return _replaceFrameworkWidget(node, name, (fw) => {
        const fwOpts = fw.options || {};
        const choices = (opts.choices || fwOpts.values || []).slice();
        const label = opts.label ?? name;
        const initialValue = fw.value;
        const origCallback = fw.callback;

        return {
            type: "MESH_PILL_COMBO",
            name: name,
            label: label,
            value: initialValue,
            options: { ...fwOpts, values: choices },
            _mesh_role: `combo_${name}`,

            draw(ctx, n, ww, y) {
                const h = SLOT_H;
                const padX = 8;
                drawPillBackground(ctx, padX, y + 2, ww - padX * 2, h - 4);

                ctx.textBaseline = "middle";
                ctx.font = "12px Segoe UI, Arial";

                ctx.fillStyle = PILL_COLORS.label;
                ctx.textAlign = "left";
                const labelText = `${this.label}:`;
                ctx.fillText(labelText, padX + 8, y + h / 2);
                const labelW = ctx.measureText(labelText).width;

                const arrowW = 16;
                const valueX = padX + 8 + labelW + 8;
                const maxW = ww - valueX - arrowW - padX - 8;
                ctx.fillStyle = PILL_COLORS.value;
                ctx.fillText(ellipsize(ctx, String(this.value ?? ""), maxW), valueX, y + h / 2);

                ctx.fillStyle = PILL_COLORS.arrow;
                ctx.textAlign = "right";
                ctx.fillText("▾", ww - padX - 6, y + h / 2);
            },

            computeSize() { return [0, SLOT_H]; },

            mouse(event, pos, n) {
                if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
                showPickerModal(this.options.values, this.value, {
                    title: `Pick ${this.label}`,
                    placeholder: "Type to filter…",
                    onSelect: (v) => {
                        this.value = v;
                        if (origCallback) {
                            try { origCallback.call(this, v); } catch (e) { /* noop */ }
                        }
                        n.setDirtyCanvas(true, true);
                    },
                });
                return true;
            },
        };
    });
}

function createPillBool(node, name, opts = {}) {
    return _replaceFrameworkWidget(node, name, (fw) => {
        const label = opts.label ?? name;
        const initialValue = !!fw.value;
        const origCallback = fw.callback;

        return {
            type: "MESH_PILL_BOOL",
            name: name,
            label: label,
            value: initialValue,
            options: fw.options || {},
            _mesh_role: `bool_${name}`,

            draw(ctx, n, ww, y) {
                const h = SLOT_H;
                const padX = 8;
                drawPillBackground(ctx, padX, y + 2, ww - padX * 2, h - 4);

                ctx.textBaseline = "middle";
                ctx.font = "12px Segoe UI, Arial";

                ctx.fillStyle = PILL_COLORS.label;
                ctx.textAlign = "left";
                ctx.fillText(`${this.label}:`, padX + 8, y + h / 2);

                // ON/OFF chip on right
                const chipW = 50;
                const chipH = h - 12;
                const chipX = ww - padX - chipW - 6;
                const chipY = y + 6;
                ctx.fillStyle = this.value ? PILL_COLORS.boolOn : PILL_COLORS.boolOff;
                ctx.fillRect(chipX, chipY, chipW, chipH);
                ctx.strokeStyle = PILL_COLORS.border;
                ctx.strokeRect(chipX + 0.5, chipY + 0.5, chipW - 1, chipH - 1);
                ctx.fillStyle = "#fff";
                ctx.textAlign = "center";
                ctx.font = "11px Segoe UI, Arial";
                ctx.fillText(this.value ? "ON" : "OFF", chipX + chipW / 2, y + h / 2);
            },

            computeSize() { return [0, SLOT_H]; },

            mouse(event, pos, n) {
                if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
                this.value = !this.value;
                if (origCallback) {
                    try { origCallback.call(this, this.value); } catch (e) { /* noop */ }
                }
                n.setDirtyCanvas(true, true);
                return true;
            },
        };
    });
}

function createPillString(node, name, opts = {}) {
    return _replaceFrameworkWidget(node, name, (fw) => {
        const label = opts.label ?? name;
        const initialValue = fw.value || "";
        const origCallback = fw.callback;

        return {
            type: "MESH_PILL_STRING",
            name: name,
            label: label,
            value: initialValue,
            options: fw.options || {},
            _mesh_role: `string_${name}`,

            draw(ctx, n, ww, y) {
                const h = SLOT_H;
                const padX = 8;
                drawPillBackground(ctx, padX, y + 2, ww - padX * 2, h - 4);

                ctx.textBaseline = "middle";
                ctx.font = "12px Segoe UI, Arial";

                ctx.fillStyle = PILL_COLORS.label;
                ctx.textAlign = "left";
                const labelText = `${this.label}:`;
                ctx.fillText(labelText, padX + 8, y + h / 2);
                const labelW = ctx.measureText(labelText).width;

                const editIconW = 20;
                const valueX = padX + 8 + labelW + 8;
                const maxW = ww - valueX - editIconW - padX - 8;
                ctx.fillStyle = PILL_COLORS.value;
                const display = String(this.value || "");
                ctx.fillText(ellipsize(ctx, display, maxW), valueX, y + h / 2);

                ctx.fillStyle = PILL_COLORS.arrow;
                ctx.textAlign = "right";
                ctx.font = "11px Segoe UI, Arial";
                ctx.fillText("✎", ww - padX - 6, y + h / 2);
            },

            computeSize() { return [0, SLOT_H]; },

            mouse(event, pos, n) {
                if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
                const entered = window.prompt(`${this.label}:`, String(this.value || ""));
                if (entered !== null) {
                    this.value = entered;
                    if (origCallback) {
                        try { origCallback.call(this, entered); } catch (e) { /* noop */ }
                    }
                    n.setDirtyCanvas(true, true);
                }
                return true;
            },
        };
    });
}

// Custom-typed clickable button. Useful for the future controls
// (confirm-restart button, status indicators, etc.) Push to the END
// of node.widgets — buttons are typically actions, not inputs, so
// positional serialization isn't a concern (buttons have no value
// that needs to round-trip through the workflow JSON).
function createMeshButton(node, role, label, onClick) {
    const widget = {
        type: "MESH_BUTTON",
        name: label,
        value: null,
        options: {},
        _mesh_role: role,
        serialize: false,  // safe here because we PUSH to end, not splice into a positional slot

        draw(ctx, n, ww, y) {
            const h = SLOT_H;
            const padX = 8;
            ctx.fillStyle = "#363636";
            ctx.fillRect(padX, y + 2, ww - padX * 2, h - 4);
            ctx.strokeStyle = PILL_COLORS.border;
            ctx.lineWidth = 1;
            ctx.strokeRect(padX + 0.5, y + 2.5, ww - padX * 2 - 1, h - 5);

            ctx.fillStyle = PILL_COLORS.value;
            ctx.font = "12px Segoe UI, Arial";
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            ctx.fillText(this.name || "", ww / 2, y + h / 2);
        },

        computeSize() { return [0, SLOT_H]; },

        mouse(event, pos, n) {
            if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
            try { onClick(); } catch (err) { console.warn("[mesh]", err); }
            return true;
        },
    };
    node.widgets.push(widget);
    return widget;
}

// =====================================================================
// Per-node setup: replace the framework widgets with pill versions.
//
// The order of these calls doesn't matter — _replaceFrameworkWidget
// reinserts each pill at the original framework widget's index, so
// widgets_values positional serialization stays stable regardless of
// what order we call them in.
// =====================================================================

function setupMeshSplitFlux(node) {
    createPillNumber(node, "n_blocks_remote", { label: "n_blocks_remote", integer: true });
    createPillString(node, "remote_host", { label: "remote_host" });
    createPillNumber(node, "remote_port",   { label: "remote_port", integer: true });
    createPillCombo(node,  "codec_mode",    { label: "codec_mode" });
    createPillNumber(node, "codec_qp",      { label: "codec_qp", integer: true });
    createPillBool(node,   "codec_lossless",{ label: "codec_lossless" });
    createPillCombo(node,  "codec_tile_dim",{ label: "codec_tile_dim" });
    createPillBool(node,   "forward_client_loras", { label: "forward_client_loras" });

    // Wider default for new nodes — the labels + pill chrome don't
    // breathe at stock width. Saved workflows restore their stored
    // size after onNodeCreated returns, so this only affects
    // newly-dropped nodes.
    const natural = node.computeSize();
    node.setSize([Math.max(natural[0], 320), natural[1]]);
}

// =====================================================================
// Banner draw hook (under the node body)
// =====================================================================

function drawBanner(node, ctx) {
    if (!node._mesh_message_text || node.flags?.collapsed) return;

    const palette = BANNER_COLORS[node._mesh_message_level] || BANNER_COLORS.info;
    const padX = 10;
    const padY = 8;
    const lineHeight = 16;
    const fontPx = 12;

    ctx.save();
    ctx.font = `${fontPx}px Segoe UI, sans-serif`;
    const lines = wrapText(ctx, node._mesh_message_text, node.size[0] - 2 * padX);
    const bannerH = padY * 2 + lines.length * lineHeight;
    const bannerY = node.size[1] + 4;

    ctx.fillStyle = palette.bg;
    ctx.beginPath();
    ctx.roundRect(0, bannerY, node.size[0], bannerH, 6);
    ctx.fill();

    ctx.fillStyle = palette.fg;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    for (let i = 0; i < lines.length; i++) {
        ctx.fillText(lines[i], padX, bannerY + padY + i * lineHeight);
    }
    ctx.restore();
}

// =====================================================================
// Extension registration
// =====================================================================

app.registerExtension({
    name: "comfyui-mesh.UI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== MESH_NODE_NAME) return;

        // Pill widget replacement on node creation. ComfyUI populates
        // the framework widgets from INPUT_TYPES BEFORE onNodeCreated
        // fires, so they're ready to be replaced here. onConfigure
        // (workflow load) runs AFTER and reads widgets_values
        // positionally — our reinserts at the same indices keep that
        // mapping correct.
        const orig_onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = orig_onNodeCreated?.apply(this, arguments);
            try {
                setupMeshSplitFlux(this);
            } catch (e) {
                console.warn("[mesh] pill setup failed; falling back to framework widgets", e);
            }
            return result;
        };

        // Banner under the node body. Painted in onDrawForeground so
        // it composites over the node frame but below any floating UI.
        const orig_onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (orig_onDrawForeground) orig_onDrawForeground.apply(this, arguments);
            drawBanner(this, ctx);
        };
    },
});

// =====================================================================
// Server-pushed messages → node banner
// =====================================================================

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
