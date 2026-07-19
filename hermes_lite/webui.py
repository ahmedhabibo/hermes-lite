"""Hermes-Lite WebUI — FastAPI server with chat interface.

Run: python -m hermes_lite.webui --port 3007
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from hermes_lite.config import get_config
from hermes_lite.orchestrator import HermesOrchestrator
from hermes_lite.orchestrator import __version__ as PKG_VERSION
from hermes_lite.router import LiteRouter

import argparse

app = FastAPI(title="Hermes-Lite WebUI", version=PKG_VERSION)

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

_sessions: dict[str, HermesOrchestrator] = {}


async def get_or_create_session(session_id: str | None = None) -> HermesOrchestrator:
    """Get or create an orchestrator session."""
    sid = session_id or f"webui-{uuid.uuid4().hex[:12]}"
    if sid not in _sessions:
        orch = HermesOrchestrator(
            db_path=f"~/.hermes_lite/webui-{sid}.db",
            session_title=f"webui:{sid}",
        )
        orch._create_default_tools()
        await orch._initialize_memory()
        _sessions[sid] = orch
    return _sessions[sid]


# ---------------------------------------------------------------------------
# WebSocket chat
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """WebSocket endpoint for streaming chat."""
    await ws.accept()
    orch = None
    try:
        # First message is session init
        init = json.loads(await ws.receive_text())
        session_id = init.get("session_id")
        orch = await get_or_create_session(session_id)
        await ws.send_text(json.dumps({"type": "connected", "session_id": orch.session_id}))

        while True:
            data = json.loads(await ws.receive_text())
            if data.get("type") == "chat":
                prompt = data.get("message", "").strip()
                if not prompt:
                    continue

                # Determine model label from config (not hardcoded strings).
                label_cfg = get_config()
                force_tier = getattr(orch, "force_tier", None)
                if force_tier == "cloud":
                    cloud_short = label_cfg.cloud_model.split("/")[-1] or label_cfg.cloud_model
                    model_label = f"cloud:{cloud_short} ☁️"
                else:
                    local_short = (
                        label_cfg.local_model.split(".")[0]
                        if label_cfg.local_model
                        else "local"
                    )
                    model_label = f"local:{local_short} ⚡"
                await ws.send_text(json.dumps({"type": "thinking", "model": model_label}))

                # Run the prompt through the orchestrator (handles /cloud, /local, /help, etc.)
                try:
                    response = await orch._handle_prompt(prompt)
                    await ws.send_text(json.dumps({
                        "type": "response",
                        "content": response,
                        "timestamp": time.time(),
                    }))
                except Exception as e:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "content": f"Error: {e}",
                        "timestamp": time.time(),
                    }))

            elif data.get("type") == "stream":
                prompt = data.get("message", "").strip()
                if not prompt:
                    continue

                # Real token streaming via orchestrator -> chat_stream.
                # Tool-requiring prompts still fall back to the full
                # batch response — they finish fast and the client sees
                # the complete reply with a stream_end signal.
                try:
                    await ws.send_text(json.dumps({"type": "stream_start"}))
                    async for token in orch.stream_prompt(prompt):
                        if token:
                            await ws.send_text(json.dumps({
                                "type": "stream_delta",
                                "content": token,
                            }))
                    await ws.send_text(json.dumps({"type": "stream_end"}))
                except Exception as e:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "content": f"Stream error: {e}",
                    }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "content": str(e)}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": app.version,
        "routing": "local-first",
    }


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": list(_sessions.keys())}


@app.post("/api/chat")
async def rest_chat(payload: dict):
    """Non-streaming chat endpoint."""
    prompt = payload.get("message", "").strip()
    session_id = payload.get("session_id")
    if not prompt:
        return JSONResponse({"error": "empty message"}, status_code=400)

    orch = await get_or_create_session(session_id)
    try:
        response = await orch._handle_prompt(prompt)
        return {"response": response, "session_id": orch.session_id}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# WebUI HTML
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hermes-Lite WebUI</title>
<style>
:root { --bg: #0d1117; --surface: #161b22; --border: #30363d; --text: #c9d1d9; --accent: #58a6ff; --green: #3fb950; --orange: #f0883e; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }
header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 20px; display: flex; align-items: center; gap: 12px; }
header .logo { font-size: 1.3rem; font-weight: bold; }
header .logo span { color: var(--accent); }
header .badge { background: var(--green); color: #000; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: bold; }
header .info { margin-left: auto; font-size: 0.8rem; color: #8b949e; }
.messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
.msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; }
.msg.user { background: #1c2128; border: 1px solid var(--border); align-self: flex-end; }
.msg.assistant { background: var(--surface); border: 1px solid var(--border); align-self: flex-start; }
.msg.assistant pre { background: #0d1117; padding: 8px 12px; border-radius: 6px; overflow-x: auto; margin-top: 8px; }
.msg.assistant code { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.9em; }
.msg.error { background: #3d1f1f; border-color: #f85149; color: #f85149; }
.typing { align-self: flex-start; color: #8b949e; font-style: italic; padding: 8px 16px; }
.typing::after { content: '●●●'; animation: dots 1.4s infinite; }
@keyframes dots { 0%,20% { opacity: 0.2; } 50% { opacity: 1; } 100% { opacity: 0.2; } }
.input-bar { background: var(--surface); border-top: 1px solid var(--border); padding: 16px 20px; display: flex; gap: 12px; }
.input-bar input { flex: 1; background: #0d1117; border: 1px solid var(--border); color: var(--text); padding: 10px 16px; border-radius: 8px; font-size: 1rem; outline: none; }
.input-bar input:focus { border-color: var(--accent); }
.input-bar button { background: var(--accent); color: #000; border: none; padding: 10px 24px; border-radius: 8px; font-size: 1rem; font-weight: bold; cursor: pointer; }
.input-bar button:disabled { opacity: 0.5; cursor: not-allowed; }
.stream-toggle { display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: #8b949e; }
.stream-toggle input { accent-color: var(--accent); }
</style>
</head>
<body>
<header>
    <div class="logo">⚡ <span>Hermes-Lite</span></div>
    <div class="badge">__HERMES_VERSION__</div>
    <div class="info" id="status">● Connecting...</div>
</header>
<div class="messages" id="messages">
    <div class="msg assistant">Welcome to Hermes-Lite <span id="version-tag">__HERMES_VERSION__</span> — local-first! ⚡ Default model: <span id="model-tag">local:Qwen2.5-Coder-7B</span>. Type <code>/cloud</code> to force cloud NIM, <code>/local</code> to return to local, <code>/help</code> for all commands.</div>
</div>
<div class="input-bar">
    <input type="text" id="prompt" placeholder="Type your message..." autocomplete="off" autofocus />
    <div class="stream-toggle"><input type="checkbox" id="stream-mode" checked /> Stream</div>
    <button id="send" onclick="sendMsg()">Send</button>
</div>
<script>
let ws = null;
let sessionId = null;
const msgsEl = document.getElementById('messages');
const inputEl = document.getElementById('prompt');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');
const streamMode = document.getElementById('stream-mode');

function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${proto}://${location.host}/ws/chat`);
    ws.onopen = () => {
        statusEl.textContent = '● Connected';
        statusEl.style.color = '#3fb950';
        ws.send(JSON.stringify({type: 'init', session_id: null}));
    };
    ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        switch(data.type) {
            case 'connected':
                sessionId = data.session_id;
                break;
            case 'thinking':
                showTyping();
                break;
            case 'stream_start':
                streamMsg = addMsg('assistant', '');
                break;
            case 'stream_delta':
                if (streamMsg) {
                    streamMsg.innerHTML += data.content.replace(/\\n/g, '<br>');
                    scrollDown();
                }
                break;
            case 'stream_end':
                hideTyping();
                streamMsg = null;
                sendBtn.disabled = false;
                break;
            case 'response':
                hideTyping();
                addMsg('assistant', data.content);
                sendBtn.disabled = false;
                break;
            case 'error':
                hideTyping();
                addMsg('error', data.content);
                sendBtn.disabled = false;
                break;
        }
    };
    ws.onclose = () => {
        statusEl.textContent = '● Disconnected';
        statusEl.style.color = '#f85149';
        setTimeout(connect, 2000);
    };
}

let streamMsg = null;

function addMsg(cls, text) {
    const div = document.createElement('div');
    div.className = `msg ${cls}`;
    div.innerHTML = text.replace(/\\n/g, '<br>');
    if (cls === 'user') {
        div.innerHTML = '❯ ' + div.innerHTML;
    }
    msgsEl.appendChild(div);
    scrollDown();
    return div;
}

function showTyping() {
    const div = document.createElement('div');
    div.className = 'typing';
    div.id = 'typing';
    div.textContent = 'Thinking';
    msgsEl.appendChild(div);
    scrollDown();
}

function hideTyping() {
    const t = document.getElementById('typing');
    if (t) t.remove();
}

function scrollDown() {
    msgsEl.scrollTop = msgsEl.scrollHeight;
}

function sendMsg() {
    const text = inputEl.value.trim();
    if (!text || !ws || ws.readyState !== 1) return;
    addMsg('user', text);
    inputEl.value = '';
    sendBtn.disabled = true;

    // Slash commands always go through the chat path (not streaming)
    // so the orchestrator can handle /cloud, /local, /model, /help, etc.
    const isCommand = text.startsWith('/') || text.startsWith('!');
    if (streamMode.checked && !isCommand) {
        ws.send(JSON.stringify({type: 'stream', message: text}));
    } else {
        ws.send(JSON.stringify({type: 'chat', message: text}));
    }
}

inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMsg();
    }
});

connect();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    cfg = get_config()
    version_tag = f"v{app.version}"
    local_tag = f"local:{cfg.local_model.split('.')[0] if cfg.local_model else ''}"
    rendered = (
        HTML_PAGE
        .replace("__HERMES_VERSION__", version_tag)
        .replace("local:Qwen2.5-Coder-7B", local_tag)
    )
    return rendered


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    from hermes_lite.config import get_config
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Hermes-Lite WebUI")
    parser.add_argument("--port", type=int, default=cfg.webui_port, help="Port to run on")
    parser.add_argument("--host", type=str, default=cfg.webui_host, help="Host to bind to")
    args = parser.parse_args()

    print(f"⚡ Hermes-Lite WebUI starting on http://{args.host}:{args.port}")
    print(f"   Tailscale: http://100.121.26.118:{args.port}")
    print(f"   Local: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
