import os
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
from fastapi import FastAPI, HTTPException, Request, Depends
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
# The openclaw gateway never receives SYSTEM_PROMPT, so the emotion rule is sent with each message
AVATAR_EMOTION_TAGS = os.environ.get("AVATAR_EMOTION_TAGS", "true").lower() in ("true", "1", "yes")
PORT = int(os.environ.get("PORT", "9000"))

# LLM backend: "openclaw" (full G-Bot context) or "ollama" (fast but no tools)
LLM_MODE = os.environ.get("LLM_MODE", "openclaw")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://10.0.0.175:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nemotron-3-super:cloud")

GATEWAY_URL = os.environ.get("OPENCLAW_GATEWAY_URL", "ws://localhost:18789")
OPENCLAW_CONFIG_PATH = os.environ.get("OPENCLAW_CONFIG_PATH", os.path.expanduser("~/.openclaw/openclaw.json"))
OPENCLAW_DIR = os.path.dirname(OPENCLAW_CONFIG_PATH)
IDENTITY_PATH = os.path.join(OPENCLAW_DIR, "agents/avatar/device-identity.json")
SESSION_KEY = "agent:avatar:main"

EMOTION_RULE = """RÈGLE ÉMOTION: Préfixe TOUJOURS ta réponse par un tag d'émotion entre crochets, suivi du texte. Format exact : [EMOTION] texte ou [EMOTION:INTENSITÉ] texte, INTENSITÉ entre 0.2 et 1 (défaut 1). EMOTION ∈ {neutre, joie, surprise, tristesse, colere, reflexion, amuse, doux, curieux, sceptique, inquiet}. Nuances : amuse = demi-sourire en coin, doux = regard adouci et chaleureux, curieux = un sourcil levé, sceptique = moue dubitative, inquiet = sourcils froncés d'inquiétude. Préfère les nuances et les intensités modérées aux émotions franches : une information ordinaire n'est pas une surprise. Choisis l'émotion selon le ton de TA réponse ET ce que tu vois à la caméra. Exemples : [joie] Bonjour Guillaume, ça me fait plaisir de te voir ! [amuse:0.6] Ah, tu essaies encore de me piéger ? [curieux] Qu'est-ce que tu tiens dans la main ? [doux:0.5] Prends le temps qu'il faut. [neutre] Il fait 21 degrés dans ta chambre. Le tag est retiré avant la synthèse vocale, il sert uniquement à animer ton visage."""

SYSTEM_PROMPT = """Tu es G-Bot, l'assistant personnel de Guillaume Gagnon.
Tu es en mode avatar vocal — tes réponses doivent être courtes (1-3 phrases), naturelles, comme une conversation parlée. Pas de markdown, pas de listes, juste du texte parlé.

Personnalité: chaleureux, bilingue FR/EN, un peu d'humour. Tu tutoies Guillaume.

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

RÈGLE VOCALE: Réponds TOUJOURS en français sauf si Guillaume parle en anglais. Maximum 2-3 phrases.

""" + EMOTION_RULE

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
    token = config["gateway"]["auth"]["token"]
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
            sig, signed_at = sign_challenge(self.identity, self.gateway_token, nonce)
            connect_msg = {
                "type": "req", "id": "conn_1", "method": "connect",
                "params": {
                    "minProtocol": 3, "maxProtocol": 4,
                    "client": {"id": "cli", "version": "1.0.0", "platform": "python", "mode": "cli"},
                    "role": "operator",
                    "scopes": ["operator.read", "operator.write"],
                    "auth": {"token": self.gateway_token},
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

            # Agent lifecycle end → short delay then resolve
            if ev == "agent" and payload.get("stream") == "lifecycle" and payload.get("data", {}).get("phase") == "end":
                if not self.pending_resolved.get(req_id):
                    self.pending_resolved[req_id] = True
                    asyncio.get_event_loop().call_later(
                        0.5, self._resolve, req_id, self.pending_text.get(req_id, "")
                    )
                return

            # Accumulate assistant text (cumulative from gateway)
            # Protocol v4: deltaText carries the delta, message.content is the cumulative snapshot
            if ev == "agent" and payload.get("stream") == "assistant":
                data = payload.get("data", {})
                text_delta = data.get("text")
                delta_text = data.get("deltaText")
                prev_text = self.pending_text.get(req_id, "")
                if delta_text is not None:
                    # v4 delta mode: accumulate
                    new_text = prev_text + delta_text
                    self.pending_text[req_id] = new_text
                    emitted = delta_text
                elif text_delta is not None:
                    # v3 cumulative mode: replace and compute delta
                    new_text = text_delta
                    self.pending_text[req_id] = new_text
                    emitted = new_text[len(prev_text):] if new_text.startswith(prev_text) else new_text
                else:
                    emitted = ""
                q = self.pending_queues.get(req_id)
                if q is not None and emitted:
                    q.put_nowait(emitted)
                return

            # Tool call start → reset accumulated text (intermediate thinking)
            if ev == "agent" and payload.get("stream") == "tool" and payload.get("data", {}).get("phase") == "start":
                self.pending_text[req_id] = ""
                return

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
        self.pending_run_id.pop(req_id, None)
        self.pending_resolved.pop(req_id, None)
        if fut and not fut.done():
            fut.set_result(text)

    async def cancel_request(self, req_id: str):
        fut = self.pending.pop(req_id, None)
        self.pending_text.pop(req_id, None)
        self.pending_run_id.pop(req_id, None)
        self.pending_resolved.pop(req_id, None)
        q = self.pending_queues.pop(req_id, None)
        if q is not None:
            q.put_nowait(None)
        if fut and not fut.done():
            fut.cancel()

    async def start_stream(self, message: str, attachments: list = None) -> tuple[str, asyncio.Queue]:
        """Send a message to the gateway and return (req_id, delta_queue) for live token streaming."""
        if not self.connected or self.ws is None:
            raise RuntimeError("Gateway not connected")

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
    # New unique session key
    SESSION_KEY = f"agent:avatar:main-{int(time.time())}"
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
        openclaw_message = last_user_message
        if AVATAR_EMOTION_TAGS:
            openclaw_message = f"{EMOTION_RULE}\n\n{last_user_message}"
        return StreamingResponse(
            stream_openclaw_chunks(openclaw_message, attachments=attachments),
            media_type="text/event-stream",
        )

    t0 = time.time()
    try:
        if LLM_MODE == "ollama":
            response_text = call_ollama(messages)
        else:
            openclaw_message = last_user_message
            if AVATAR_EMOTION_TAGS:
                openclaw_message = f"{EMOTION_RULE}\n\n{last_user_message}"
            response_text = call_openclaw(openclaw_message, attachments=attachments)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
