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

// Horizontal padding between the node's outer border and the pill body.
// Same value on both sides → pills are centered within the node body.
// Bumped from 8 to give a visible margin so widgets don't hug the node
// edge, especially on the right where the framework's reserved socket
// area used to make pills look left-aligned.
const MESH_PILL_PAD = 22;

// =====================================================================
// Last-used values (globally remembered across fresh node drops)
//
// ComfyUI already persists widget values per-workflow via the saved
// JSON. This adds a second layer: when the user drops a FRESH node
// (no saved values to restore from), pre-fill it with whatever the
// last-touched mesh node was set to — so they don't have to retype
// remote_host / n_blocks_remote / etc every time.
//
// Stored in browser localStorage as a single JSON object keyed by
// widget name. Saved on every value change. Loaded on node creation
// AFTER the pill widgets are in place; if onConfigure fires later
// (workflow restore), it overwrites our last-used with the saved
// values, which is correct — the workflow's intent wins.
// =====================================================================

const LAST_USED_KEY = "comfyui-mesh.lastUsed.MeshSplitFlux";

function _loadLastUsed() {
    try {
        const raw = localStorage.getItem(LAST_USED_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        return (parsed && typeof parsed === "object") ? parsed : {};
    } catch (e) {
        return {};
    }
}

function _saveLastUsed(values) {
    try {
        localStorage.setItem(LAST_USED_KEY, JSON.stringify(values));
    } catch (e) { /* noop — localStorage may be disabled */ }
}

function _captureNodeValues(node) {
    const out = {};
    for (const w of (node.widgets || [])) {
        if (w._mesh_role && w.value !== undefined && w.value !== null) {
            out[w.name] = w.value;
        }
    }
    return out;
}

function _persistLastUsed(node) {
    if (!node) return;
    _saveLastUsed(_captureNodeValues(node));
}

// =====================================================================
// Confirm-restart button: shown when n_blocks_remote drifts from its
// baseline (the value at last node load or last successful reconfigure).
// Click POSTs to /mesh/reconfigure, which causes the server to execv
// itself with the new --n-blocks and the client to transparently
// reconnect to the freshly-restarted process.
// =====================================================================

function _checkPendingState(node) {
    if (!node) return;
    const nbW = node.widgets?.find((w) => w.name === "n_blocks_remote");
    const baseline = node._mesh_baseline_n_blocks;
    const target = nbW?.value;
    const isPending =
        nbW != null && baseline != null && target != null && target !== baseline;
    const btn = node._mesh_confirm_btn;
    if (btn) {
        btn._mesh_visible = isPending;
        btn._mesh_target = target;
        // Hidden/visible changes the widget's reported height, so a
        // resize is needed to make the layout re-flow.
        const computed = node.computeSize();
        node.setSize([Math.max(node.size[0], computed[0]), computed[1]]);
    }
    node.setDirtyCanvas(true, true);
}

function _onValueChange(node, widget) {
    _persistLastUsed(node);
    if (widget && widget.name === "n_blocks_remote") {
        _checkPendingState(node);
    }
}

// =====================================================================
// Always-on connection indicator (bottom of the node)
//
// Polls /mesh/status every CONN_POLL_MS. Renders as a flat row at the
// bottom of the node (no pill chrome, just a colored circle + text +
// host:port). State semantics match the Python route:
//
//   connected    — cached MeshClient socket is live; data can flow
//   disconnected — cached client socket died (server crashed, network)
//   idle         — no MeshClient yet; user hasn't queued for this
//                  host:port this session
//
// The polling interval is per-node; cleaned up on node.onRemoved.
// =====================================================================

const CONN_POLL_MS = 3000;

const CONN_COLORS = {
    connected:    { dot: "#3b9b3b", text: "Connected" },
    disconnected: { dot: "#c43030", text: "Disconnected" },
    idle:         { dot: "#888888", text: "Idle" },
};

function createConnectionIndicator(node) {
    const widget = {
        type: "MESH_CONN_INDICATOR",
        name: "_mesh_conn",
        value: null,
        options: {},
        serialize: false,
        _mesh_role: "conn_indicator",
        _mesh_state: "idle",
        _mesh_n_blocks: null,

        draw(ctx, n, ww, y) {
            const W = n.size[0];
            const h = SLOT_H * 0.7;  // slimmer than a pill
            const padX = MESH_PILL_PAD;

            ctx.textBaseline = "middle";
            ctx.font = "11px Segoe UI, Arial";

            const conf = CONN_COLORS[this._mesh_state] || CONN_COLORS.idle;

            // Status dot
            const cx = padX + 8;
            const cy = y + h / 2;
            const r = 5;
            ctx.fillStyle = conf.dot;
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "#1a1a1a";
            ctx.lineWidth = 1;
            ctx.stroke();

            // Status label
            ctx.fillStyle = "#ddd";
            ctx.textAlign = "left";
            ctx.fillText(conf.text, cx + r + 8, cy);

            // Right side: host:port (and server n_blocks if known)
            const hostW = n.widgets.find((w) => w.name === "remote_host");
            const portW = n.widgets.find((w) => w.name === "remote_port");
            if (hostW && portW) {
                let right = `${hostW.value}:${portW.value}`;
                if (this._mesh_n_blocks != null) {
                    right += `  ·  server n=${this._mesh_n_blocks}`;
                }
                ctx.fillStyle = "#888";
                ctx.textAlign = "right";
                ctx.fillText(right, W - padX - 8, cy);
            }
        },

        computeSize() { return [0, Math.round(SLOT_H * 0.7)]; },

        // Not interactive.
        mouse() { return false; },
    };
    node.widgets.push(widget);
    return widget;
}

// =====================================================================
// Help / troubleshooting modal
//
// Opened by the "❓" button at the bottom of the node. Shows a few
// categories of tips covering the most common surprises: connection,
// silent-wrong-output causes, LoRA ordering, the decrease-needs-
// restart story, and the ComfyUI-version-mismatch gotcha (which we
// can't auto-detect — server hello_ack doesn't carry the version).
// =====================================================================

function _helpHTML() {
    return `
<style>
  .mesh-help h3 { margin: 14px 0 4px; font-size: 13px; color: #9bf; font-weight: 600; }
  .mesh-help h3:first-child { margin-top: 0; }
  .mesh-help ul { margin: 4px 0 6px; padding-left: 20px; color: #ddd; }
  .mesh-help li { margin: 3px 0; }
  .mesh-help code { background: #1a1a1a; padding: 1px 5px; border-radius: 3px;
                    font-size: 12px; color: #ffe; }
  .mesh-help em { color: #ffc; font-style: normal; font-weight: 600; }
</style>
<div class="mesh-help">

<h3>🔌 Connection</h3>
<ul>
  <li>Indicator at the bottom of the node: <em>green</em> = connected,
      <em>red</em> = disconnected (server died or network gone),
      <em>grey</em> = idle (no queue this session yet).</li>
  <li>Refused / never connects: check the server is running
      (run <code>run_server_flux2_gui.bat</code> on the back-half host),
      the <code>remote_host</code> + <code>remote_port</code> match,
      and the server's port isn't blocked by a firewall.</li>
  <li>Server died mid-session: just re-queue. Transparent reconnect
      handles it — no need to relaunch ComfyUI.</li>
</ul>

<h3>🎚 Mismatched n_blocks (silent wrong output prevention)</h3>
<ul>
  <li>The Confirm button (orange) appears when <code>n_blocks_remote</code>
      drifts from what the server is currently running. Click it to
      restart the server with the new value before you queue.</li>
  <li><em>Increasing</em> <code>n_blocks_remote</code> is seamless:
      Confirm restarts the server; the client's strip extends
      incrementally. No ComfyUI restart needed.</li>
  <li><em>Decreasing</em> requires a ComfyUI restart on this side —
      the client-side stripped block weights are gone for the session
      and can only be reloaded from disk by a fresh ComfyUI launch.
      The inline banner under the node will tell you when this
      applies.</li>
</ul>

<h3>🎨 LoRAs</h3>
<ul>
  <li>Workflow ordering matters: <code>LoraLoader</code> must come
      <em>BEFORE</em> Icarus in the graph for the LoRA to be
      visible to this node and forwarded to the server.</li>
  <li>Keep <code>forward_client_loras</code> ON so the LoRA also
      affects back-half blocks (the ones running on the server).</li>
  <li>The server can also load its own LoRA at startup (GUI option) —
      it stacks with whatever the client forwards.</li>
</ul>

<h3>⚡ Performance / quality</h3>
<ul>
  <li><code>codec_qp</code>: 18 (default) is sharp. Towards 28 the
      image gets noticeably softer with visible noise. 10 is
      near-lossless.</li>
  <li><code>codec_tile_dim</code>: leave at 4. Higher = fewer larger
      NVENC frames per encode = faster wall-clock; 4 is the sweet spot.</li>
  <li>Same machine (two GPUs)? Set <code>codec_mode</code> to
      <em>raw</em>. PCIe between two GPUs is faster than NVENC
      encode/decode — codec only helps on slow wires (LAN, VPN,
      residential broadband).</li>
</ul>

<h3>🧩 ComfyUI version mismatch (silent-correctness gotcha)</h3>
<ul>
  <li>The server runs its own ComfyUI clone (in the server folder's
      <code>..\\ComfyUI</code>). The fp8 detection + FLUX implementation
      evolve in upstream over time; a big drift between the version
      this client uses and the version the server uses can produce
      subtly wrong output with no error.</li>
  <li>Fix on the server host: run <code>update_comfy.bat</code> in the
      server folder. <code>git pull</code>s ComfyUI + re-installs its
      requirements.</li>
  <li>This node doesn't know the server's ComfyUI version (the wire
      protocol doesn't carry it), so we can't warn you automatically
      — keep both ends reasonably current to avoid drift.</li>
</ul>

<h3>📦 Files</h3>
<ul>
  <li>Server install: <code>install.bat</code> in the server folder
      (one-shot — venv + ComfyUI clone + cu128 torch + deps).</li>
  <li>Server update: <code>update_comfy.bat</code> (the one above).</li>
  <li>Server launch: <code>run_server_flux2_gui.bat</code> (recommended) or
      <code>run_server_flux2.bat</code> (headless).</li>
</ul>

<h3>💬 Help / feedback</h3>
<ul>
  <li>Bug / feature request: see the project's README for contact
      details.</li>
  <li>If this rig saves you a GPU and you'd like more model
      architectures supported (Wan, LTX-Video, FLUX.1, SD3.5 …),
      <code>buymeacoffee.com/lorasandlenses</code> — community demand
      drives priority.</li>
</ul>
</div>
`;
}

function showHelpModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = `
        position: fixed; inset: 0; background: rgba(0,0,0,0.6);
        display: flex; align-items: center; justify-content: center;
        z-index: 10000; font-family: Segoe UI, sans-serif;
    `;
    const modal = document.createElement("div");
    modal.style.cssText = `
        background: #2a2a2a; color: #ddd; border: 1px solid #555;
        border-radius: 8px; padding: 16px 20px; width: 660px;
        max-width: 92vw; max-height: 80vh; display: flex;
        flex-direction: column; gap: 10px;
    `;

    const header = document.createElement("div");
    header.style.cssText =
        "font-size: 15px; font-weight: 600; color: #fff; padding-bottom: 4px; " +
        "border-bottom: 1px solid #444;";
    header.textContent = "ComfyUI Mesh : Icarus — Tips & troubleshooting";

    const body = document.createElement("div");
    body.style.cssText =
        "overflow-y: auto; flex: 1; font-size: 13px; line-height: 1.5; " +
        "padding-right: 4px;";
    body.innerHTML = _helpHTML();

    const footer = document.createElement("div");
    footer.style.cssText = "display: flex; justify-content: flex-end; padding-top: 4px;";
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "Close";
    closeBtn.style.cssText = `
        background: #444; color: #ddd; border: 1px solid #555;
        padding: 6px 16px; border-radius: 4px; cursor: pointer;
        font-family: inherit; font-size: 13px;
    `;
    footer.appendChild(closeBtn);

    modal.appendChild(header);
    modal.appendChild(body);
    modal.appendChild(footer);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);

    const close = () => backdrop.remove();
    closeBtn.addEventListener("click", close);
    backdrop.addEventListener("click", (e) => {
        if (e.target === backdrop) close();
    });
    const escHandler = (e) => {
        if (e.key === "Escape") {
            close();
            document.removeEventListener("keydown", escHandler);
        }
    };
    document.addEventListener("keydown", escHandler);
}

function startConnectionPoll(node) {
    let stopped = false;

    const tick = async () => {
        if (stopped) return;
        const indicator = node._mesh_conn_indicator;
        if (!indicator) return;
        const hostW = node.widgets.find((w) => w.name === "remote_host");
        const portW = node.widgets.find((w) => w.name === "remote_port");
        if (!hostW || !portW) return;
        try {
            const url = `/mesh/status?host=${encodeURIComponent(String(hostW.value))}` +
                        `&port=${encodeURIComponent(String(portW.value))}`;
            const resp = await fetch(url);
            if (resp.ok) {
                const data = await resp.json();
                const state = (data && data.state) || "idle";
                const newN = (data && data.server_n_blocks != null)
                    ? data.server_n_blocks : null;
                if (indicator._mesh_state !== state ||
                    indicator._mesh_n_blocks !== newN) {
                    indicator._mesh_state = state;
                    indicator._mesh_n_blocks = newN;
                    node.setDirtyCanvas(true, true);
                }
            }
        } catch (e) {
            // Network error reaching ComfyUI itself — show as disconnected
            // so the user gets a useful signal.
            if (indicator._mesh_state !== "disconnected") {
                indicator._mesh_state = "disconnected";
                node.setDirtyCanvas(true, true);
            }
        }
    };

    tick();  // initial check, no wait
    const interval = setInterval(tick, CONN_POLL_MS);

    const orig_onRemoved = node.onRemoved;
    node.onRemoved = function () {
        stopped = true;
        clearInterval(interval);
        if (orig_onRemoved) orig_onRemoved.apply(this, arguments);
    };
}

function createConfirmButton(node) {
    // Pushed to the END of node.widgets so it never participates in
    // positional widgets_values serialization (serialize:false too,
    // for triple-safety). Hidden by default — computeSize returns a
    // zero-ish height when _mesh_visible is false, so the slot
    // collapses cleanly.
    const widget = {
        type: "MESH_CONFIRM_BTN",
        name: "_mesh_confirm",
        value: null,
        options: {},
        serialize: false,
        _mesh_role: "confirm_btn",
        _mesh_visible: false,
        _mesh_target: null,
        _mesh_in_flight: false,

        draw(ctx, n, ww, y) {
            if (!this._mesh_visible) return;
            const W = n.size[0];
            const h = SLOT_H;
            const padX = MESH_PILL_PAD;
            const inFlight = this._mesh_in_flight;

            ctx.fillStyle = inFlight ? "#665030" : "#a06530";
            ctx.fillRect(padX, y + 2, W - padX * 2, h - 4);
            ctx.strokeStyle = "#c08850";
            ctx.lineWidth = 1;
            ctx.strokeRect(padX + 0.5, y + 2.5, W - padX * 2 - 1, h - 5);

            ctx.fillStyle = "#fff";
            ctx.font = "bold 12px Segoe UI, Arial";
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            const target = this._mesh_target ?? "?";
            const label = inFlight
                ? `Restarting server with n_blocks=${target}…`
                : `✓ Confirm: restart server with n_blocks=${target}`;
            ctx.fillText(label, W / 2, y + h / 2);
        },

        computeSize() {
            return this._mesh_visible ? [0, SLOT_H] : [0, -4];
        },

        mouse(event, pos, n) {
            if (!this._mesh_visible || this._mesh_in_flight) return false;
            if (event.type !== "pointerdown" && event.type !== "mousedown") return false;

            const hostW = n.widgets.find((w) => w.name === "remote_host");
            const portW = n.widgets.find((w) => w.name === "remote_port");
            const nbW   = n.widgets.find((w) => w.name === "n_blocks_remote");
            if (!hostW || !portW || !nbW) return true;

            this._mesh_in_flight = true;
            n.setDirtyCanvas(true, true);

            fetch("/mesh/reconfigure", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    host: String(hostW.value || ""),
                    port: parseInt(portW.value, 10) || 0,
                    n_blocks: parseInt(nbW.value, 10) || 0,
                }),
            }).then(async (resp) => {
                this._mesh_in_flight = false;
                if (resp.ok) {
                    // Reconfigure succeeded — update baseline and clear
                    // any pending banner, and hide ourselves.
                    n._mesh_baseline_n_blocks = nbW.value;
                    this._mesh_visible = false;
                    delete n._mesh_message_text;
                    delete n._mesh_message_level;
                } else {
                    let detail = "";
                    try { detail = (await resp.json())?.error || ""; } catch (e) {}
                    n._mesh_message_text =
                        `Reconfigure failed: ${detail || resp.statusText || resp.status}`;
                    n._mesh_message_level = "warn";
                }
                const computed = n.computeSize();
                n.setSize([Math.max(n.size[0], computed[0]), computed[1]]);
                n.setDirtyCanvas(true, true);
            }).catch((err) => {
                this._mesh_in_flight = false;
                n._mesh_message_text = `Reconfigure call failed: ${err}`;
                n._mesh_message_level = "warn";
                n.setDirtyCanvas(true, true);
            });
            return true;
        },
    };
    node.widgets.push(widget);
    return widget;
}

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
                // Use n.size[0] directly — the `ww` LiteGraph passes is
                // narrower than the node body on some ComfyUI builds,
                // which made pills look left-clustered with empty space
                // on the right.
                const W = n.size[0];
                const h = SLOT_H;
                const padX = MESH_PILL_PAD;
                const arrowW = 16;
                drawPillBackground(ctx, padX, y + 2, W - padX * 2, h - 4);

                ctx.textBaseline = "middle";
                ctx.font = "12px Segoe UI, Arial";

                ctx.fillStyle = PILL_COLORS.arrow;
                ctx.textAlign = "center";
                ctx.fillText("◀", padX + arrowW / 2 + 4, y + h / 2);
                ctx.fillText("▶", W - padX - arrowW / 2 - 4, y + h / 2);

                ctx.fillStyle = PILL_COLORS.label;
                ctx.textAlign = "left";
                const labelX = padX + arrowW + 12;
                ctx.fillText(this.label, labelX, y + h / 2);

                ctx.fillStyle = PILL_COLORS.value;
                ctx.textAlign = "right";
                const valStr = this.options.integer
                    ? String(Math.round(this.value))
                    : this.value.toFixed(2);
                ctx.fillText(valStr, W - padX - arrowW - 12, y + h / 2);
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
                const W = n.size[0];
                const padX = MESH_PILL_PAD;
                const arrowHit = 24;
                const onLeft = x < padX + arrowHit;
                const onRight = x > W - padX - arrowHit;

                if (event.type === "pointerdown" || event.type === "mousedown") {
                    if (onLeft) {
                        this._setValue(this.value - this.options.step);
                        n.setDirtyCanvas(true, true);
                        _onValueChange(n, this);
                        return true;
                    }
                    if (onRight) {
                        this._setValue(this.value + this.options.step);
                        n.setDirtyCanvas(true, true);
                        _onValueChange(n, this);
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
                                _onValueChange(n, this);
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
                        _onValueChange(n, this);
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
                const W = n.size[0];
                const h = SLOT_H;
                const padX = MESH_PILL_PAD;
                drawPillBackground(ctx, padX, y + 2, W - padX * 2, h - 4);

                ctx.textBaseline = "middle";
                ctx.font = "12px Segoe UI, Arial";

                ctx.fillStyle = PILL_COLORS.label;
                ctx.textAlign = "left";
                const labelText = `${this.label}:`;
                ctx.fillText(labelText, padX + 8, y + h / 2);
                const labelW = ctx.measureText(labelText).width;

                const arrowW = 16;
                const valueX = padX + 8 + labelW + 8;
                const maxW = W - valueX - arrowW - padX - 8;
                ctx.fillStyle = PILL_COLORS.value;
                ctx.fillText(ellipsize(ctx, String(this.value ?? ""), maxW), valueX, y + h / 2);

                ctx.fillStyle = PILL_COLORS.arrow;
                ctx.textAlign = "right";
                ctx.fillText("▾", W - padX - 6, y + h / 2);
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
                        _onValueChange(n, this);
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
                const W = n.size[0];
                const h = SLOT_H;
                const padX = MESH_PILL_PAD;
                drawPillBackground(ctx, padX, y + 2, W - padX * 2, h - 4);

                ctx.textBaseline = "middle";
                ctx.font = "12px Segoe UI, Arial";

                ctx.fillStyle = PILL_COLORS.label;
                ctx.textAlign = "left";
                ctx.fillText(`${this.label}:`, padX + 8, y + h / 2);

                // ON/OFF chip on right
                const chipW = 50;
                const chipH = h - 12;
                const chipX = W - padX - chipW - 6;
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
                _persistLastUsed(n);
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
                const W = n.size[0];
                const h = SLOT_H;
                const padX = MESH_PILL_PAD;
                drawPillBackground(ctx, padX, y + 2, W - padX * 2, h - 4);

                ctx.textBaseline = "middle";
                ctx.font = "12px Segoe UI, Arial";

                ctx.fillStyle = PILL_COLORS.label;
                ctx.textAlign = "left";
                const labelText = `${this.label}:`;
                ctx.fillText(labelText, padX + 8, y + h / 2);
                const labelW = ctx.measureText(labelText).width;

                const editIconW = 20;
                const valueX = padX + 8 + labelW + 8;
                const maxW = W - valueX - editIconW - padX - 8;
                ctx.fillStyle = PILL_COLORS.value;
                const display = String(this.value || "");
                ctx.fillText(ellipsize(ctx, display, maxW), valueX, y + h / 2);

                ctx.fillStyle = PILL_COLORS.arrow;
                ctx.textAlign = "right";
                ctx.font = "11px Segoe UI, Arial";
                ctx.fillText("✎", W - padX - 6, y + h / 2);
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
                    _persistLastUsed(n);
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
            const W = n.size[0];
            const h = SLOT_H;
            const padX = MESH_PILL_PAD;
            ctx.fillStyle = "#363636";
            ctx.fillRect(padX, y + 2, W - padX * 2, h - 4);
            ctx.strokeStyle = PILL_COLORS.border;
            ctx.lineWidth = 1;
            ctx.strokeRect(padX + 0.5, y + 2.5, W - padX * 2 - 1, h - 5);

            ctx.fillStyle = PILL_COLORS.value;
            ctx.font = "12px Segoe UI, Arial";
            ctx.textBaseline = "middle";
            ctx.textAlign = "center";
            ctx.fillText(this.name || "", W / 2, y + h / 2);
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

// Pill widget chrome (label + value + chevrons + ON/OFF chip) needs
// breathing room. Anything narrower clips the labels or overflows the
// node bounds. Enforced as both:
//   1. New-node default width (in setupMeshSplitFlux)
//   2. Hard minimum on the node prototype's computeSize (so loaded
//      workflows that saved a narrower size get bumped up too)
const MESH_NODE_MIN_W = 410;

function setupMeshSplitFlux(node) {
    // Pass min/max/step EXPLICITLY for INT pills — fw.options has been
    // unreliable across ComfyUI versions (missing min/max on some
    // builds, default step that isn't 1 on others). Hardcoding here
    // matches the values declared in the Python INPUT_TYPES.
    createPillNumber(node, "n_blocks_remote", {
        label: "n_blocks_remote", integer: true, min: 0, max: 256, step: 1,
    });
    createPillString(node, "remote_host", { label: "remote_host" });
    createPillNumber(node, "remote_port", {
        label: "remote_port", integer: true, min: 1, max: 65535, step: 1,
    });
    createPillCombo(node, "codec_mode", { label: "codec_mode" });
    createPillNumber(node, "codec_qp", {
        label: "codec_qp", integer: true, min: 0, max: 51, step: 1,
    });
    createPillBool(node,   "codec_lossless",{ label: "codec_lossless" });
    createPillCombo(node,  "codec_tile_dim",{ label: "codec_tile_dim" });
    createPillBool(node,   "forward_client_loras", { label: "forward_client_loras" });

    // Apply globally remembered last-used values. Loaded workflows
    // will overwrite this in onConfigure (which fires AFTER us with
    // the saved widgets_values), so saved-workflow intent always wins
    // — last-used only matters for fresh node drops.
    const lastUsed = _loadLastUsed();
    for (const w of node.widgets) {
        if (w._mesh_role && Object.prototype.hasOwnProperty.call(lastUsed, w.name)) {
            const v = lastUsed[w.name];
            // Defensive: only restore if the type roughly matches what
            // the widget already holds (e.g. don't poke a number into a
            // string field if localStorage is corrupted).
            if (typeof v === typeof w.value || w.value === null || w.value === undefined) {
                w.value = v;
            }
        }
    }

    // Snapshot n_blocks_remote AFTER last-used has been applied. This
    // is the baseline the Confirm button compares against. onConfigure
    // (workflow load) refreshes it later if a saved value gets restored.
    const nbW = node.widgets.find((w) => w.name === "n_blocks_remote");
    node._mesh_baseline_n_blocks = nbW ? nbW.value : null;

    // Confirm-restart button (hidden by default).
    node._mesh_confirm_btn = createConfirmButton(node);

    // Always-on connection indicator at the bottom of the node.
    node._mesh_conn_indicator = createConnectionIndicator(node);
    startConnectionPoll(node);

    // Help / tips button at the very bottom. Opens a modal with
    // categorised troubleshooting (connection, mismatched n_blocks,
    // LoRA ordering, perf, version-mismatch gotcha, etc.).
    createMeshButton(node, "help_btn", "❓ Help / tips & troubleshooting", showHelpModal);

    // Default width for freshly-dropped nodes. computeSize() (overridden
    // below in beforeRegisterNodeDef) already enforces MESH_NODE_MIN_W
    // as a floor, so the Math.max here is belt-and-braces.
    const natural = node.computeSize();
    node.setSize([Math.max(natural[0], MESH_NODE_MIN_W), natural[1]]);
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

        // Enforce minimum width as a floor on computeSize. LiteGraph
        // calls this any time it needs to know the natural/minimum size
        // (initial layout, widget add/remove, restoring from saved
        // workflows). Returning a width of MESH_NODE_MIN_W means the
        // node never settles narrower than what the pill chrome needs,
        // even if a previously-saved workflow stored a smaller size.
        const orig_computeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function () {
            const sz = orig_computeSize ? orig_computeSize.apply(this, arguments) : [200, 80];
            return [Math.max(sz[0], MESH_NODE_MIN_W), sz[1]];
        };

        // Workflow-load path: ComfyUI restores node.size + widgets_values
        // from the saved JSON AFTER onNodeCreated. If the saved size
        // predates pill widgets (or the user manually shrank the node),
        // the restored width can leave the pills clipped. Bump up to
        // the minimum here so loaded workflows look right too.
        //
        // Also: snapshot the loaded n_blocks_remote as the Confirm
        // button's baseline — anything saved IS the in-sync state by
        // definition, so the button should be hidden until the user
        // changes it.
        const orig_onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = orig_onConfigure?.apply(this, arguments);
            if (this.size && this.size[0] < MESH_NODE_MIN_W) {
                this.setSize([MESH_NODE_MIN_W, this.size[1]]);
            }
            const nbW = this.widgets?.find((w) => w.name === "n_blocks_remote");
            if (nbW) {
                this._mesh_baseline_n_blocks = nbW.value;
            }
            if (this._mesh_confirm_btn) {
                this._mesh_confirm_btn._mesh_visible = false;
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
