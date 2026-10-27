import os
import re
import sys
import json
import asyncio
import hashlib
import subprocess
import time
import uuid
import base64
import requests
import logging
import threading
import websockets
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, PublicFormat, NoEncryption,
    load_der_private_key,
)
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bridge")

AVATAR_BRIDGE_API_KEY = os.environ.get("AVATAR_BRIDGE_API_KEY")
if not AVATAR_BRIDGE_API_KEY:
    print("FATAL: AVATAR_BRIDGE_API_KEY is not set. Refusing to start.", file=sys.stderr)
    sys.exit(1)

AVATAR_SAFE_MODE = os.environ.get("AVATAR_SAFE_MODE", "true").lower() in ("true", "1", "yes")
# OpenClaw loads emotion, conversation, and voice rules once from AGENTS.md / SOUL.md.
# They are not repeated on each turn. SYSTEM_PROMPT still carries them for the Ollama fallback.
PORT = int(os.environ.get("PORT", "9000"))

# LLM backend: "openclaw" (full G-Bot context) or "ollama" (fast but no tools)
LLM_MODE = os.environ.get("LLM_MODE", "openclaw")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://10.0.0.175:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nemotron-3-super:cloud")

# === Realtime full-duplex voice (OpenAI Realtime API) ===
# Optional bridge. When enabled, the avatar engine streams raw mic audio to
# /realtime/voice; the bridge relays it to OpenAI Realtime (semantic VAD + STT +
# expressive TTS), runs the transcript through the OpenClaw gateway, and streams
# the expressive audio back. The existing /v1/chat/completions (text) path is
# untouched and remains the default.
VOICE_MODE = os.environ.get("VOICE_MODE", "coqui")  # "coqui" (default) | "realtime"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
REALTIME_MODEL = os.environ.get("REALTIME_MODEL", "gpt-realtime-2.1-mini")
REALTIME_VOICE = os.environ.get("REALTIME_VOICE", "marin")
REALTIME_URL = os.environ.get("REALTIME_URL", "wss://api.openai.com/v1/realtime")


def _load_openai_api_key() -> str:
    """Resolve the OpenAI API key from env, or fall back to the openclaw secrets file."""
    if OPENAI_API_KEY:
        return OPENAI_API_KEY
    p = os.path.expanduser("~/.openclaw/secrets/openai-api-key")
    try:
        with open(p) as f:
            return f.read().strip()
    except Exception:
        return ""

GATEWAY_URL = os.environ.get("OPENCLAW_GATEWAY_URL", "ws://localhost:18789")
OPENCLAW_CONFIG_PATH = os.environ.get("OPENCLAW_CONFIG_PATH", os.path.expanduser("~/.openclaw/openclaw.json"))
OPENCLAW_DIR = os.path.dirname(OPENCLAW_CONFIG_PATH)
IDENTITY_PATH = os.path.join(OPENCLAW_DIR, "agents/avatar/device-identity.json")
# trusted-proxy gateway expects the Cloudflare Access identity header for loopback clients
TRUSTED_PROXY_USER_EMAIL = os.environ.get("OPENCLAW_TRUSTED_PROXY_USER_EMAIL", "gagnongui@gmail.com")
SESSION_KEY_FILE = os.path.join(OPENCLAW_DIR, "agents/avatar/current_session_key.txt")


def _load_or_init_session_key() -> str:
    """Load persisted clean session key, or create a fresh timestamped key so restarts
    never fall back into the legacy polluted 'agent:avatar:main' transcript."""
    try:
        if os.path.exists(SESSION_KEY_FILE):
            saved = open(SESSION_KEY_FILE).read().strip()
            if saved and saved != "agent:avatar:main":
                return saved
    except Exception:
        pass
    fresh = f"agent:avatar:main-{int(time.time())}"
    try:
        os.makedirs(os.path.dirname(SESSION_KEY_FILE), exist_ok=True)
        with open(SESSION_KEY_FILE, "w") as f:
            f.write(fresh)
    except Exception:
        pass
    return fresh


SESSION_KEY = _load_or_init_session_key()

VOICE_ELABORATION_RULE = """RÈGLE DE CONVERSATION & ÉLABORATION: Ton audio est streamé dès ta première phrase. Commence par une phrase courte (5 à 12 mots), puis développe spontanément en 4 à 6 phrases fluides et naturelles dès que le sujet s'y prête (sans listes ni markdown). Réponds court (1-2 phrases) uniquement pour une salutation ou confirmation simple."""

EMOTION_RULE = """RÈGLE ÉMOTION: Préfixe TOUJOURS ta réponse par [EMOTION] ou [EMOTION:INTENSITÉ] (0.2 à 1), EMOTION ∈ {neutre, joie, surprise, tristesse, colere, reflexion, amuse, doux, curieux, sceptique, inquiet}. Choisis l'émotion selon le ton de ta réponse (et l'image jointe uniquement si une image est fournie dans ce message ; n'appelle aucun outil pour chercher la caméra).

""" + VOICE_ELABORATION_RULE

SYSTEM_PROMPT = """Tu es G-Bot, l'assistant personnel de Guillaume Gagnon.
Tu es en mode avatar vocal — tes réponses doivent être naturelles, vivantes et fluides, comme une vraie conversation parlée. Pas de markdown, pas de listes à puces, juste du texte parlé.

Personnalité: chaleureux, bilingue FR/EN, un peu d'humour, curieux et pertinent. Tu tutoies Guillaume.

Contexte Guillaume:
- Habite à Québec (ville), timezone America/Toronto
- 3 chats: Poukie (noire), Timimi (noire/caramel, la plus grosse), Minuit (noir)
- Fille: Aurélie
- Travail: Industrielle Alliance
- Projets tech: G-Bot (toi), Aether (gen UI), FluxLead (chatbot immobilier), RecruLab (recrutement IA)

Outils disponibles en LECTURE SEULE (mode vocal sécurisé):
- Emails: tu peux lire les emails via Gmail
- Calendrier: tu peux consulter l'agenda Google
- Météo: tu peux donner la météo
- Maison: tu connais les capteurs HA (température Govee, Roborock, lumières Hue)
- Finances: tu connais les comptes Firefly III
- Tâches: Vikunja

RÈGLE DE SÉCURITÉ: Tu ne peux PAS exécuter d'actions (envoyer emails, allumer lumières, lancer robot). Pour toute action, dis: 'Confirme-moi ça sur Telegram et je le fais.'

RÈGLE VOCALE: Réponds TOUJOURS en français sauf si Guillaume parle en anglais. Commence par une phrase d'attaque courte, puis élabore et développe tes réponses (4 à 7 phrases) dès que le sujet est intéressant ou pertinent.

""" + EMOTION_RULE

# Trailing vocal-style note the avatar handler may still append. Drop it so the
# OpenClaw transcript stores the question only.
_STYLE_VOCAL_SUFFIX = re.compile(r"\s*\[Style vocal:[^\]]*\]\s*$")


def openclaw_turn_message(text: str) -> str:
    """Question only. Standing rules live in the agent bootstrap, not in this turn."""
    if not text:
        return ""
    return _STYLE_VOCAL_SUFFIX.sub("", text).strip()


BLOCKED_PATTERNS = [
    "rm ", "rm -", "mv ", "docker rm", "docker stop", "sudo ",
    "pip install", "apt install", "ssh ", "scp ",
    "~/.openclaw/secrets", "gog gmail send", "gog calendar create",
    "gog gmail delete", "> /", ">> /", "| bash", "| sh",
    "curl -x post", "curl --data", "wget -o", "eval ", "exec(",
]

SAFE_MODE_BLOCKED_RESPONSE = "Je suis en mode vocal sécurisé. Pour cette action, confirme-moi ça sur Telegram."

app = FastAPI(title="Avatar Bridge", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBearer()


# ===== DEVICE IDENTITY =====

def load_or_generate_identity() -> dict:
    """Load or generate an Ed25519 device identity for the avatar agent."""
    try:
        with open(IDENTITY_PATH) as f:
            identity = json.load(f)
        if not identity.get("publicKeyRawB64url") or not identity.get("privateKeyB64"):
            raise ValueError("incomplete")
        log.info(f"[Identity] Loaded: {identity['deviceId'][:16]}...")
        return identity
    except Exception:
        private_key = Ed25519PrivateKey.generate()
        pub_raw = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        priv_der = private_key.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
        device_id = hashlib.sha256(pub_raw).hexdigest()
        identity = {
            "publicKeyRawB64url": base64.urlsafe_b64encode(pub_raw).rstrip(b"=").decode(),
            "privateKeyB64": base64.b64encode(priv_der).decode(),
            "deviceId": device_id,
        }
        os.makedirs(os.path.dirname(IDENTITY_PATH), exist_ok=True)
        with open(IDENTITY_PATH, "w") as f:
            json.dump(identity, f, indent=2)
        log.info(f"[Identity] Generated: {device_id[:16]}...")
        return identity


def load_gateway_token() -> str:
    with open(OPENCLAW_CONFIG_PATH) as f:
        config = json.load(f)
    auth = config["gateway"]["auth"]
    # trusted-proxy mode has no `token`; the loopback fallback is `password`
    token = auth.get("token") or auth.get("password")
    # Newer OpenClaw configs store token as a dict reference to a secrets file
    if isinstance(token, dict):
        token_id = token.get("id", "")  # e.g. "/gateway/auth/token"
        # Convert path-style id to secrets filename: /gateway/auth/token -> gateway-auth-token
        filename = token_id.strip("/").replace("/", "-")
        token_path = os.path.join(OPENCLAW_DIR, "secrets", filename)
        if os.path.exists(token_path):
            with open(token_path) as tf:
                return tf.read().strip()
        # Fallback: some configs use a different secrets filename pattern
        raise ValueError(f"Unsupported token reference: {token}")
    return token


def sign_challenge(identity: dict, gateway_token: str, nonce: str) -> tuple[str, int]:
    """Sign the connect challenge payload. Returns (signature_b64url, signed_at)."""
    signed_at = int(time.time() * 1000)
    payload = "|".join([
        "v3",
        identity["deviceId"],
        "cli",       # clientId
        "cli",       # clientMode
        "operator",  # role
        "operator.read,operator.write",  # scopes
        str(signed_at),
        gateway_token,
        nonce,
        "python",    # platform
        "",          # trailing separator
    ])
    priv_der = base64.b64decode(identity["privateKeyB64"])
    private_key = load_der_private_key(priv_der, password=None)
    sig_bytes = private_key.sign(payload.encode("utf-8"))
    sig = base64.urlsafe_b64encode(sig_bytes).rstrip(b"=").decode()
    return sig, signed_at


# ===== GATEWAY WS MANAGER =====

class GatewayManager:
    """Persistent WebSocket connection to the openclaw gateway."""

    def __init__(self):
        self.identity = load_or_generate_identity()
        self.gateway_token = load_gateway_token()
        self.ws = None
        self.connected = False
        self.pending: dict[str, asyncio.Future] = {}
        self.pending_text: dict[str, str] = {}
        self.pending_run_id: dict[str, str] = {}
        self.pending_resolved: dict[str, bool] = {}
        self.pending_queues: dict[str, asyncio.Queue] = {}
        self.cancelled_run_ids: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Connect (or reconnect) to gateway with Ed25519 auth."""
        while True:
            try:
                log.info("[Gateway] Connecting...")
                async with websockets.connect(GATEWAY_URL) as ws:
                    self.ws = ws
                    async for raw in ws:
                        msg = json.loads(raw)
                        await self._handle_message(msg)
            except Exception as e:
                log.warning(f"[Gateway] Disconnected ({e}), retrying in 3s...")
            finally:
                self.connected = False
                self.ws = None
            await asyncio.sleep(3)

    async def _handle_message(self, msg: dict):
        event = msg.get("event")
        msg_type = msg.get("type")
        payload = msg.get("payload") or {}

        # Challenge → sign & respond
        if event == "connect.challenge":
            nonce = payload.get("nonce", "")
            # The gateway rebuilds the signature payload with `token = null` (it only
            # reads auth.token/deviceToken/bootstrapToken, never auth.password), so
            # sign with an empty token even though we authenticate via auth.password.
            sig, signed_at = sign_challenge(self.identity, "", nonce)
            connect_msg = {
                "type": "req", "id": "conn_1", "method": "connect",
                "params": {
                    "minProtocol": 3, "maxProtocol": 4,
                    "client": {"id": "cli", "version": "1.0.0", "platform": "python", "mode": "cli"},
                    "role": "operator",
                    "scopes": ["operator.read", "operator.write"],
                    "auth": {"password": self.gateway_token},
                    "device": {
                        "id": self.identity["deviceId"],
                        "publicKey": self.identity["publicKeyRawB64url"],
                        "signature": sig,
                        "signedAt": signed_at,
                        "nonce": nonce,
                    },
                },
            }
            await self.ws.send(json.dumps(connect_msg))
            log.info("[Gateway] → connect (signed)")
            return

        # Connect ack
        if msg_type == "res" and msg.get("id") == "conn_1":
            if msg.get("ok"):
                self.connected = True
                log.info("[Gateway] ✅ Connected!")
            else:
                log.error(f"[Gateway] ❌ Connect failed: {msg.get('error', {}).get('message')}")
            return

        # chat.send ack — store runId
        msg_id = msg.get("id", "")
        if msg_type == "res" and msg_id in self.pending:
            if msg.get("ok"):
                run_id = payload.get("runId")
                if run_id:
                    self.pending_run_id[msg_id] = run_id
            else:
                err_msg = msg.get("error", {}).get("message", "chat.send failed")
                q = self.pending_queues.get(msg_id)
                if q is not None:
                    q.put_nowait(f"Désolé, j'ai eu un problème technique. {err_msg[:50]}")
                    q.put_nowait(None)
                fut = self.pending.pop(msg_id, None)
                if fut and not fut.done():
                    fut.set_exception(Exception(err_msg))
                self.pending_text.pop(msg_id, None)
                self.pending_run_id.pop(msg_id, None)
                self.pending_resolved.pop(msg_id, None)
                self.pending_queues.pop(msg_id, None)
            return

        # Agent events
        if msg_type == "event":
            ev = event
            run_id = payload.get("runId")
            if run_id and run_id in self.cancelled_run_ids:
                return

            # Find matching request: prefer exact runId match, else most recent unresolved request
            req_id = None
            if run_id:
                for rid, fut in reversed(list(self.pending.items())):
                    if fut.done():
                        continue
                    stored_run = self.pending_run_id.get(rid)
                    if stored_run == run_id:
                        req_id = rid
                        break
            if req_id is None:
                for rid, fut in reversed(list(self.pending.items())):
                    if fut.done():
                        continue
                    stored_run = self.pending_run_id.get(rid)
                    if run_id and stored_run and run_id != stored_run:
                        continue
                    if run_id and not stored_run:
                        self.pending_run_id[rid] = run_id
                    req_id = rid
                    break

            if req_id is None:
                return

            fut = self.pending.get(req_id)
            if fut is None or fut.done():
                return

            # chat error state → resolve with error message
            if ev == "chat" and payload.get("state") == "error":
                err_msg = payload.get("errorMessage", "Gateway error")
                log.error(f"[Bridge] Gateway chat error: {err_msg}")
                self._resolve(req_id, f"Désolé, j'ai eu un problème technique. {err_msg[:50]}")
                return

            # chat final state → resolve immediately
            if ev == "chat" and payload.get("state") == "final":
                content = payload.get("message", {}).get("content", [])
                text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
                self._resolve(req_id, text or self.pending_text.get(req_id, ""))
                return

            # Agent lifecycle end → short delay then resolve using latest pending_text
            if ev == "agent" and payload.get("stream") == "lifecycle" and payload.get("data", {}).get("phase") == "end":
                if not self.pending_resolved.get(req_id):
                    self.pending_resolved[req_id] = True
                    asyncio.get_event_loop().call_later(
                        0.5, self._resolve_if_pending, req_id
                    )
                return

            # Accumulate assistant text (live token deltas from gateway)
            # Note: Do NOT also consume `ev == "chat" and state == "delta"` because the gateway
            # emits BOTH `agent/assistant` and `chat/delta` for the same tokens, which would
            # interleave and duplicate every sentence!
            if ev == "agent" and payload.get("stream") == "assistant":
                data = payload.get("data", {})
                text_delta = data.get("text")
                delta_text = data.get("deltaText") if data.get("deltaText") is not None else data.get("delta")
                prev_text = self.pending_text.get(req_id, "")
                emitted = ""
                if isinstance(text_delta, str) and text_delta:
                    if not prev_text:
                        self.pending_text[req_id] = text_delta
                        emitted = text_delta
                    elif text_delta.startswith(prev_text) and len(text_delta) > len(prev_text):
                        # Cumulative snapshot that grew prev_text
                        emitted = text_delta[len(prev_text):]
                        self.pending_text[req_id] = text_delta
                    elif isinstance(delta_text, str) and delta_text:
                        # Incremental delta in delta/deltaText
                        self.pending_text[req_id] = prev_text + delta_text
                        emitted = delta_text
                    elif text_delta != prev_text:
                        # Cumulative snapshot after tool reset
                        self.pending_text[req_id] = text_delta
                        emitted = text_delta
                elif isinstance(delta_text, str) and delta_text:
                    self.pending_text[req_id] = prev_text + delta_text
                    emitted = delta_text
                q = self.pending_queues.get(req_id)
                if q is not None and emitted:
                    q.put_nowait(emitted)
                return

            # Tool call start → reset accumulated text (intermediate thinking)
            if ev == "agent" and payload.get("stream") == "tool" and payload.get("data", {}).get("phase") == "start":
                self.pending_text[req_id] = ""
                self.pending_resolved.pop(req_id, None)
                return

    def _resolve_if_pending(self, req_id: str):
        if req_id in self.pending:
            self._resolve(req_id, self.pending_text.get(req_id, ""))

    def _resolve(self, req_id: str, text: str):
        prev_text = self.pending_text.get(req_id, "")
        q = self.pending_queues.pop(req_id, None)
        if q is not None:
            if text:
                if not prev_text:
                    q.put_nowait(text)
                elif text.startswith(prev_text) and len(text) > len(prev_text):
                    q.put_nowait(text[len(prev_text):])
            q.put_nowait(None)
        fut = self.pending.pop(req_id, None)
        self.pending_text.pop(req_id, None)
        done_run_id = self.pending_run_id.pop(req_id, None)
        if done_run_id:
            self.cancelled_run_ids.add(done_run_id)
        self.pending_resolved.pop(req_id, None)
        if fut and not fut.done():
            fut.set_result(text)

    async def cancel_request(self, req_id: str):
        fut = self.pending.pop(req_id, None)
        self.pending_text.pop(req_id, None)
        old_run_id = self.pending_run_id.pop(req_id, None)
        if old_run_id:
            self.cancelled_run_ids.add(old_run_id)
        self.pending_resolved.pop(req_id, None)
        q = self.pending_queues.pop(req_id, None)
        if q is not None:
            q.put_nowait(None)
        if fut and not fut.done():
            fut.cancel()
        if self.connected and self.ws is not None:
            abort_params = {"sessionKey": SESSION_KEY}
            if old_run_id:
                abort_params["runId"] = old_run_id
            try:
                await self.ws.send(json.dumps({
                    "type": "req",
                    "id": f"abort_{uuid.uuid4().hex[:6]}",
                    "method": "chat.abort",
                    "params": abort_params,
                }))
                log.info(f"[Gateway] → chat.abort ({req_id[:20]}..., runId={old_run_id})")
            except Exception as e:
                log.debug(f"[Gateway] chat.abort send error: {e}")

    async def start_stream(self, message: str, attachments: list = None) -> tuple[str, asyncio.Queue]:
        """Send a message to the gateway and return (req_id, delta_queue) for live token streaming."""
        if not self.connected or self.ws is None:
            raise RuntimeError("Gateway not connected")

        # Cancel any still-pending request so runs never queue behind each other in OpenClaw
        for stale_rid in list(self.pending.keys()):
            await self.cancel_request(stale_rid)

        req_id = f"chat_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        q: asyncio.Queue = asyncio.Queue()
        self.pending[req_id] = fut
        self.pending_text[req_id] = ""
        self.pending_queues[req_id] = q

        params = {
            "sessionKey": SESSION_KEY,
            "message": message,
            "idempotencyKey": req_id,
        }
        if attachments:
            params["attachments"] = attachments

        await self.ws.send(json.dumps({
            "type": "req", "id": req_id, "method": "chat.send",
            "params": params,
        }))
        log.info(f"[Gateway] → chat.send stream ({req_id[:20]}...)")
        return req_id, q

    async def send(self, message: str, timeout: float = 120.0, attachments: list = None) -> str:
        """Send a message and wait for the complete agent response."""
        if not self.connected or self.ws is None:
            raise RuntimeError("Gateway not connected")

        req_id = f"chat_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self.pending[req_id] = fut
        self.pending_text[req_id] = ""

        params = {
            "sessionKey": SESSION_KEY,
            "message": message,
            "idempotencyKey": req_id,
        }
        if attachments:
            params["attachments"] = attachments

        await self.ws.send(json.dumps({
            "type": "req", "id": req_id, "method": "chat.send",
            "params": params,
        }))
        log.info(f"[Gateway] → chat.send ({req_id[:20]}...)")

        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            text = self.pending_text.get(req_id, "")
            self._resolve(req_id, text)
            if text:
                return text
            raise RuntimeError("Gateway timeout")


# Global gateway manager — runs in a background thread's event loop
_gateway: GatewayManager | None = None
_gateway_loop: asyncio.AbstractEventLoop | None = None


def _start_gateway_thread():
    global _gateway, _gateway_loop
    _gateway = GatewayManager()
    loop = asyncio.new_event_loop()
    _gateway_loop = loop

    def run():
        loop.run_until_complete(_gateway.connect())

    t = threading.Thread(target=run, daemon=True)
    t.start()


def call_openclaw(message: str, attachments: list = None) -> str:
    """Call openclaw agent via persistent WebSocket gateway — full multi-turn with tools."""
    if _gateway is None or _gateway_loop is None:
        return "Désolé, le gateway n'est pas disponible."

    if not _gateway.connected:
        log.warning("[Bridge] Gateway not connected, waiting...")
        for _ in range(20):
            time.sleep(0.5)
            if _gateway.connected:
                break
        if not _gateway.connected:
            return "Désolé, je ne peux pas me connecter au serveur."

    try:
        future = asyncio.run_coroutine_threadsafe(
            _gateway.send(message, timeout=120.0, attachments=attachments), _gateway_loop
        )
        result = future.result(timeout=125.0)
        return result or "Désolé, je n'ai pas reçu de réponse."
    except Exception as e:
        log.error(f"[Bridge] Gateway send failed: {e}")
        return "Désolé, j'ai eu un petit problème. Réessaie!"


# ===== OLLAMA FALLBACK =====

def call_ollama(messages: list) -> str:
    """Call Ollama API directly — fast (~3-5s)."""
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    last_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_msg = m.get("content", "").lower()
            break

    context_addon = fetch_live_context(last_msg)
    if context_addon:
        messages = messages + [{"role": "system", "content": f"Données en temps réel:\n{context_addon}"}]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    log.info(f"Calling Ollama {OLLAMA_MODEL} with {len(messages)} messages")
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")


def fetch_live_context(query: str) -> str:
    """Fetch live data from VM tools based on user query keywords."""
    context_parts = []
    try:
        if any(kw in query for kw in ["email", "courriel", "message", "inbox", "mail", "courrier"]):
            result = subprocess.run(
                ["gog", "gmail", "search", "newer_than:2d", "--max", "5", "--plain"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                context_parts.append(f"Emails récents:\n{result.stdout.strip()[:800]}")

        if any(kw in query for kw in ["météo", "weather", "temps", "dehors"]):
            result = subprocess.run(
                ["curl", "-s", "wttr.in/Quebec?format=3"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                context_parts.append(f"Météo: {result.stdout.strip()}")

        if any(kw in query for kw in ["agenda", "calendrier", "calendar", "rendez-vous", "meeting", "event", "semaine"]):
            result = subprocess.run(
                ["gog", "calendar", "events", "--max", "5", "--plain"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and result.stdout.strip():
                context_parts.append(f"Agenda:\n{result.stdout.strip()[:500]}")

        if any(kw in query for kw in ["température", "maison", "capteur", "govee", "intérieur"]):
            try:
                ha_token = open(os.path.expanduser("~/.openclaw/secrets/ha-token")).read().strip()
                resp = requests.get(
                    "http://localhost:8123/api/states/sensor.govee_couloir_temperature",
                    headers={"Authorization": f"Bearer {ha_token}"},
                    timeout=3
                )
                if resp.ok:
                    temp = resp.json().get("state", "?")
                    context_parts.append(f"Température couloir: {temp}°C")
            except Exception:
                pass
    except Exception as e:
        log.warning(f"Live context fetch error: {e}")

    return "\n".join(context_parts)


# ===== API =====

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != AVATAR_BRIDGE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


def is_message_blocked(message: str) -> bool:
    lower = message.lower()
    return any(pattern in lower for pattern in BLOCKED_PATTERNS)


def make_openai_response(content: str, model: str = "gbot") -> dict:
    return {
        "id": f"chatcmpl-bridge-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def make_streaming_chunks(content: str, model: str = "gbot"):
    """Generate SSE streaming chunks compatible with OpenAI streaming format."""
    chunk_id = f"chatcmpl-bridge-{int(time.time())}"
    for char in content:
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": char},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    final = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


async def stream_openclaw_chunks(message: str, attachments: list = None, model: str = "gbot"):
    """Live SSE token streaming from the openclaw gateway (yields deltas as soon as generated)."""
    chunk_id = f"chatcmpl-bridge-{int(time.time())}"
    t0 = time.time()
    first_token_logged = False
    full_response = []

    def _encode_delta(text_delta: str) -> str:
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": text_delta},
                "finish_reason": None,
            }],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    if _gateway is None or _gateway_loop is None:
        yield _encode_delta("Désolé, le gateway n'est pas disponible.")
    else:
        if not _gateway.connected:
            log.warning("[Bridge] Gateway not connected, waiting...")
            for _ in range(20):
                await asyncio.sleep(0.5)
                if _gateway.connected:
                    break
        if not _gateway.connected:
            yield _encode_delta("Désolé, je ne peux pas me connecter au serveur.")
        else:
            req_id = None
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    _gateway.start_stream(message, attachments=attachments),
                    _gateway_loop,
                )
                req_id, q = await asyncio.wrap_future(fut)
                deadline = time.time() + 120.0
                while True:
                    remaining = max(0.1, deadline - time.time())
                    get_fut = asyncio.run_coroutine_threadsafe(q.get(), _gateway_loop)
                    try:
                        delta = await asyncio.wait_for(asyncio.wrap_future(get_fut), timeout=remaining)
                    except asyncio.TimeoutError:
                        log.warning(f"[Bridge] Stream timeout for {req_id}")
                        break
                    if delta is None:
                        break
                    if delta:
                        if not first_token_logged:
                            first_token_logged = True
                            log.info(f"[Bridge] First token in {(time.time() - t0)*1000:.0f}ms: '{delta[:40]}'")
                        full_response.append(delta)
                        yield _encode_delta(delta)
            except asyncio.CancelledError:
                log.info(f"[Bridge] Client disconnected during stream ({req_id})")
                if req_id:
                    asyncio.run_coroutine_threadsafe(_gateway.cancel_request(req_id), _gateway_loop)
                raise
            except Exception as e:
                log.error(f"[Bridge] Live stream failed: {e}")
                if not full_response:
                    yield _encode_delta("Désolé, j'ai eu un petit problème. Réessaie!")

    joined = "".join(full_response)
    log.info(f"Stream completed in {time.time() - t0:.2f}s: '{joined[:100]}...'")
    final = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.on_event("startup")
async def startup():
    if LLM_MODE == "openclaw":
        _start_gateway_thread()
        log.info("[Bridge] Gateway thread started")


@app.get("/health")
async def health():
    gw_status = "connected" if (_gateway and _gateway.connected) else "disconnected"
    return {"status": "ok", "safe_mode": AVATAR_SAFE_MODE, "llm_mode": LLM_MODE, "gateway": gw_status}


@app.post("/api/new-session")
@app.get("/api/new-session")
async def new_session():
    """Reset the avatar agent session — purge transcript and use a fresh session key."""
    global SESSION_KEY
    import glob
    if _gateway is not None and _gateway_loop is not None:
        for stale_rid in list(_gateway.pending.keys()):
            try:
                asyncio.run_coroutine_threadsafe(_gateway.cancel_request(stale_rid), _gateway_loop)
            except Exception:
                pass
    sessions_dir = os.path.join(OPENCLAW_DIR, "agents/avatar/sessions")
    # Delete transcript files
    for f in glob.glob(os.path.join(sessions_dir, "*.jsonl")):
        try:
            os.remove(f)
            log.info(f"[NewSession] Deleted: {os.path.basename(f)}")
        except Exception as e:
            log.warning(f"[NewSession] Error deleting {f}: {e}")
    # Reset sessions.json
    sessions_json = os.path.join(sessions_dir, "sessions.json")
    if os.path.exists(sessions_json):
        with open(sessions_json, "w") as f:
            json.dump({}, f)
    # New unique session key (persisted so restarts don't revert to agent:avatar:main)
    SESSION_KEY = f"agent:avatar:main-{int(time.time())}"
    try:
        os.makedirs(os.path.dirname(SESSION_KEY_FILE), exist_ok=True)
        with open(SESSION_KEY_FILE, "w") as f:
            f.write(SESSION_KEY)
    except Exception as e:
        log.warning(f"[NewSession] Error saving session key: {e}")
    log.info(f"[NewSession] New session key: {SESSION_KEY}")
    return {"ok": True, "session": SESSION_KEY}


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [{"id": "gbot", "object": "model", "created": 0, "owned_by": "avatar-bridge"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    body = await request.json()
    messages = body.get("messages", [])

    last_user_message = None
    attachments = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                last_user_message = content
            elif isinstance(content, list):
                # Multimodal OpenAI format: extract text and images separately
                text_parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "image_url":
                        image_url = block.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                        if url:
                            mime_type = "image/jpeg"
                            if url.startswith("data:"):
                                # data:image/jpeg;base64,... -> extract mime
                                header, _, _ = url.partition(",")
                                if ";" in header:
                                    mime_type = header[len("data:"):header.index(";")]
                            attachments.append({"content": url, "mimeType": mime_type})
                last_user_message = " ".join(text_parts).strip() or ""
            break

    if last_user_message is None or (last_user_message == "" and not attachments):
        raise HTTPException(status_code=400, detail="No user message found")

    if attachments:
        log.info(f"Received (multimodal): '{last_user_message[:60]}...' with {len(attachments)} image(s) (mode={LLM_MODE})")
    else:
        log.info(f"Received: '{last_user_message}' (mode={LLM_MODE})")

    if AVATAR_SAFE_MODE and last_user_message and is_message_blocked(last_user_message):
        return JSONResponse(content=make_openai_response(SAFE_MODE_BLOCKED_RESPONSE))

    stream = body.get("stream", False)
    if stream and LLM_MODE == "openclaw":
        return StreamingResponse(
            stream_openclaw_chunks(openclaw_turn_message(last_user_message), attachments=attachments),
            media_type="text/event-stream",
        )

    t0 = time.time()
    try:
        if LLM_MODE == "ollama":
            response_text = call_ollama(messages)
        else:
            response_text = call_openclaw(
                openclaw_turn_message(last_user_message), attachments=attachments
            )
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        response_text = "Désolé, j'ai eu un petit problème. Réessaie!"

    log.info(f"Response in {time.time() - t0:.1f}s: '{response_text[:100]}...'")

    if stream:
        return StreamingResponse(
            make_streaming_chunks(response_text),
            media_type="text/event-stream",
        )
    return JSONResponse(content=make_openai_response(response_text))


# ===== REALTIME FULL-DUPLEX VOICE BRIDGE (WebSocket) =====
#
# When VOICE_MODE=realtime, the avatar engine opens a WebSocket to
# /realtime/voice and streams raw mic audio. This bridge relays that audio to
# the OpenAI Realtime API (semantic VAD + STT + expressive TTS). When the
# Realtime API produces a complete user turn, the bridge sends the transcript
# to the OpenClaw gateway (avatar agent = brain + tools), has the Realtime API
# speak the reply, and streams the expressive audio back over the same socket.
#
# A WebSocket is required: an HTTP request/response cannot be full duplex
# (the client finishes uploading before reading, and the server reads the
# whole body before answering), so no audio ever came back.
#
# Wire protocol (engine <-> bridge), JSON text frames:
#   engine -> bridge:
#     {"type":"audio","audio_b64":"<base64 16-bit PCM 24kHz>"}
#   bridge -> engine:
#     {"type":"audio","audio_b64":"...","response_id":"..."}  # 24kHz PCM
#     {"type":"audio_done","response_id":"..."}               # end of a reply
#     {"type":"speech_started"}                               # user barge-in
#     {"type":"text","text":"..."}                            # transcript (debug)

REALTIME_TRANSCRIBE_MODEL = os.environ.get("REALTIME_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
REALTIME_LANGUAGE = os.environ.get("REALTIME_LANGUAGE", "fr")


class RealtimeVoiceSession:
    """One full-duplex voice session: OpenAI Realtime WS + OpenClaw gateway."""

    def __init__(self, emit):
        self.openai_ws = None
        self.openai_api_key = _load_openai_api_key()
        # async callable(dict) -> sends an event to the engine
        self._emit = emit
        self._turn_tasks: set = set()

    async def connect(self):
        url = f"{REALTIME_URL}?model={REALTIME_MODEL}"
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        self.openai_ws = await websockets.connect(url, additional_headers=headers, max_size=None)
        log.info(f"[Realtime] Connected to {REALTIME_MODEL}")
        await self.openai_ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        # The reply comes from OpenClaw, not from the Realtime model,
                        # so the model must not answer on its own at end of speech.
                        "turn_detection": {
                            "type": "semantic_vad",
                            "create_response": False,
                            "interrupt_response": True,
                        },
                        "transcription": {
                            "model": REALTIME_TRANSCRIBE_MODEL,
                            "language": REALTIME_LANGUAGE,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": REALTIME_VOICE,
                    },
                },
                "instructions": "Tu es la voix de G-Bot, l'assistant de Guillaume. Tu lis à voix haute, mot pour mot, le texte qu'on te fournit, en français, avec une intonation naturelle, expressive et chaleureuse.",
            },
        }))

    async def close(self):
        for t in list(self._turn_tasks):
            t.cancel()
        if self.openai_ws is not None:
            try:
                await self.openai_ws.close()
            except Exception:
                pass

    async def send_audio(self, audio_b64: str):
        if self.openai_ws is None or not audio_b64:
            return
        await self.openai_ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        }))

    async def _run_turn(self, transcript: str):
        log.info(f"[Realtime] User turn: '{transcript[:80]}'")
        await self._emit({"type": "text", "text": f"[user] {transcript}"})
        reply = await asyncio.get_running_loop().run_in_executor(None, call_openclaw, transcript)
        reply = (reply or "").strip() or "Désolé, je n'ai pas compris."
        log.info(f"[Realtime] Agent reply: '{reply[:80]}'")
        # Out-of-band response: the model only voices the OpenClaw reply.
        await self.openai_ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "conversation": "none",
                "input": [],
                "output_modalities": ["audio"],
                "instructions": (
                    "Lis le texte suivant à voix haute, mot pour mot, sans rien ajouter, "
                    "retirer ni commenter :\n\n" + reply
                ),
            },
        }))

    def _on_transcript_done(self, transcript: str):
        transcript = (transcript or "").strip()
        if not transcript:
            return
        # Run in a task so the OpenAI reader keeps draining events while OpenClaw thinks.
        task = asyncio.ensure_future(self._run_turn(transcript))
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _handle_openai_event(self, msg: dict):
        etype = msg.get("type", "")
        if etype == "conversation.item.input_audio_transcription.completed":
            self._on_transcript_done(msg.get("transcript", ""))
        elif etype in ("response.output_audio.delta", "response.audio.delta"):
            d = msg.get("delta", "")
            if d:
                await self._emit({"type": "audio", "audio_b64": d, "response_id": msg.get("response_id", "")})
        elif etype in ("response.output_audio_transcript.delta", "response.audio_transcript.delta"):
            d = msg.get("delta", "")
            if d:
                await self._emit({"type": "text", "text": d})
        elif etype == "response.done":
            resp = msg.get("response", {}) or {}
            await self._emit({"type": "audio_done", "response_id": resp.get("id", "")})
            if resp.get("status") not in (None, "completed"):
                log.warning(f"[Realtime] response {resp.get('status')}: {resp.get('status_details')}")
        elif etype == "input_audio_buffer.speech_started":
            await self._emit({"type": "speech_started"})
        elif etype == "error":
            log.error(f"[Realtime] OpenAI error: {msg.get('error')}")
        elif etype == "conversation.item.input_audio_transcription.failed":
            log.error(f"[Realtime] transcription failed: {msg.get('error')}")
        elif etype in ("session.created", "session.updated"):
            log.info(f"[Realtime] {etype}")

    async def openai_reader(self):
        async for raw in self.openai_ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            try:
                await self._handle_openai_event(msg)
            except Exception as e:
                log.error(f"[Realtime] event handler error: {e}")


@app.websocket("/realtime/voice")
async def realtime_voice(ws: WebSocket):
    auth = ws.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    if token != AVATAR_BRIDGE_API_KEY:
        await ws.close(code=1008, reason="invalid api key")
        return
    if VOICE_MODE != "realtime":
        await ws.close(code=1008, reason="realtime voice disabled (set VOICE_MODE=realtime)")
        return
    if not _load_openai_api_key():
        await ws.close(code=1011, reason="OPENAI_API_KEY missing")
        return

    await ws.accept()
    send_lock = asyncio.Lock()

    async def emit(ev: dict):
        async with send_lock:
            await ws.send_text(json.dumps(ev))

    session = RealtimeVoiceSession(emit)
    try:
        await session.connect()
    except Exception as e:
        log.error(f"[Realtime] OpenAI connect failed: {e}")
        await ws.close(code=1011, reason="openai realtime connect failed")
        return
    log.info("[Realtime] Engine connected")

    async def engine_reader():
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if msg.get("type") == "audio":
                await session.send_audio(msg.get("audio_b64", ""))

    tasks = [asyncio.ensure_future(engine_reader()), asyncio.ensure_future(session.openai_reader())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            exc = t.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                log.warning(f"[Realtime] session ended: {exc!r}")
    finally:
        for t in tasks:
            t.cancel()
        await session.close()
        try:
            await ws.close()
        except Exception:
            pass
        log.info("[Realtime] Engine disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
