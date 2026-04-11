import os
import sys
import json
import subprocess
import time
import requests
import logging
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bridge")

AVATAR_BRIDGE_API_KEY = os.environ.get("AVATAR_BRIDGE_API_KEY")
if not AVATAR_BRIDGE_API_KEY:
    print("FATAL: AVATAR_BRIDGE_API_KEY is not set. Refusing to start.", file=sys.stderr)
    sys.exit(1)

AVATAR_SAFE_MODE = os.environ.get("AVATAR_SAFE_MODE", "true").lower() in ("true", "1", "yes")
OPENCLAW_SESSION_ID = os.environ.get("OPENCLAW_TELEGRAM_ID") or os.environ.get("OPENCLAW_SESSION_ID", "main")
PORT = int(os.environ.get("PORT", "9000"))

# LLM backend: "openclaw" (full G-Bot context) or "ollama" (fast but no tools)
LLM_MODE = os.environ.get("LLM_MODE", "openclaw")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://10.0.0.175:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nemotron-3-super:cloud")

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

RÈGLE VOCALE: Réponds TOUJOURS en français sauf si Guillaume parle en anglais. Maximum 2-3 phrases."""

# SSH mode for openclaw
SSH_HOST = os.environ.get("OPENCLAW_SSH_HOST", "")
SSH_USER = os.environ.get("OPENCLAW_SSH_USER", "guillaume")
SSH_KEY = os.environ.get("OPENCLAW_SSH_KEY", "")

BLOCKED_PATTERNS = [
    "rm ", "rm -", "mv ", "docker rm", "docker stop", "sudo ",
    "pip install", "apt install", "ssh ", "scp ",
    "~/.openclaw/secrets", "gog gmail send", "gog calendar create",
    "gog gmail delete", "> /", ">> /", "| bash", "| sh",
    "curl -x post", "curl --data", "wget -o", "eval ", "exec(",
]

SAFE_MODE_BLOCKED_RESPONSE = "Je suis en mode vocal sécurisé. Pour cette action, confirme-moi ça sur Telegram."

app = FastAPI(title="Avatar Bridge", version="1.1.0")
security = HTTPBearer()


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
    """Generate SSE streaming chunks compatible with OpenAI streaming format.
    Sends content character by character to let the LLM handler accumulate
    and the TTS handler split by sentence naturally."""
    chunk_id = f"chatcmpl-bridge-{int(time.time())}"
    # Send content character by character — the LLM handler accumulates
    # and the TTS handler splits on sentence boundaries (. ! ?)
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
    # Send final chunk
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


def call_ollama(messages: list) -> str:
    """Call Ollama API directly — fast (~3-5s)."""
    # Add system prompt if not present
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    # Check if user is asking for live data we can fetch
    last_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_msg = m.get("content", "").lower()
            break

    context_addon = fetch_live_context(last_msg)
    if context_addon:
        messages = messages + [{"role": "system", "content": f"Donn\u00e9es en temps r\u00e9el:\n{context_addon}"}]

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
        # Bridge runs on the VM — call tools locally, no SSH needed
        if any(kw in query for kw in ["email", "courriel", "message", "inbox", "mail"]):
            result = subprocess.run(
                ["gog", "gmail", "list", "--max", "5"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                context_parts.append(f"Emails r\u00e9cents:\n{result.stdout.strip()[:500]}")

        if any(kw in query for kw in ["m\u00e9t\u00e9o", "weather", "temps", "dehors"]):
            result = subprocess.run(
                ["curl", "-s", "wttr.in/Quebec?format=3"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                context_parts.append(f"M\u00e9t\u00e9o: {result.stdout.strip()}")

        if any(kw in query for kw in ["agenda", "calendrier", "calendar", "rendez-vous", "meeting"]):
            result = subprocess.run(
                ["gog", "calendar", "list", "--max", "5"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                context_parts.append(f"Agenda:\n{result.stdout.strip()[:500]}")

        if any(kw in query for kw in ["temp\u00e9rature", "maison", "capteur", "govee", "int\u00e9rieur"]):
            try:
                ha_token = open(os.path.expanduser("~/.openclaw/secrets/ha-token")).read().strip()
                resp = requests.get(
                    "http://localhost:8123/api/states/sensor.govee_couloir_temperature",
                    headers={"Authorization": f"Bearer {ha_token}"},
                    timeout=3
                )
                if resp.ok:
                    temp = resp.json().get("state", "?")
                    context_parts.append(f"Temp\u00e9rature couloir: {temp}\u00b0C")
            except Exception:
                pass
    except Exception as e:
        log.warning(f"Live context fetch error: {e}")

    return "\n".join(context_parts)


def call_openclaw(message: str) -> str:
    """Call OpenClaw agent CLI — slow (~30-60s) but full G-Bot context."""
    try:
        if SSH_HOST:
            ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
            if SSH_KEY:
                ssh_cmd += ["-i", SSH_KEY]
            ssh_cmd += [
                f"{SSH_USER}@{SSH_HOST}",
                f"openclaw agent --to {OPENCLAW_SESSION_ID} --message {json.dumps(message)} --json"
            ]
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=120)
        else:
            result = subprocess.run(
                ["openclaw", "agent", "--agent", "avatar", "--session-id", "avatar-voice", "--thinking", "off", "--message", message, "--json"],
                capture_output=True, text=True, timeout=120,
            )
    except subprocess.TimeoutExpired:
        return "Désolé, le temps de réponse a été trop long."
    except Exception as e:
        return f"Erreur: {str(e)}"

    if result.returncode != 0:
        return "Désolé, je n'ai pas pu traiter ta demande."

    try:
        raw_output = result.stdout.strip()
        openclaw_data = None
        for line in reversed(raw_output.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    openclaw_data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        if openclaw_data is None:
            openclaw_data = json.loads(raw_output)

        if "result" in openclaw_data:
            payloads = openclaw_data["result"].get("payloads", [])
            if payloads:
                return payloads[0].get("text", "")
        return openclaw_data.get("response") or openclaw_data.get("text") or str(openclaw_data)
    except json.JSONDecodeError:
        return result.stdout.strip()


@app.get("/health")
async def health():
    return {"status": "ok", "safe_mode": AVATAR_SAFE_MODE, "llm_mode": LLM_MODE}


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
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_message = msg.get("content", "")
            break

    if last_user_message is None:
        raise HTTPException(status_code=400, detail="No user message found")

    log.info(f"Received: '{last_user_message}' (mode={LLM_MODE})")

    if AVATAR_SAFE_MODE and is_message_blocked(last_user_message):
        return JSONResponse(content=make_openai_response(SAFE_MODE_BLOCKED_RESPONSE))

    t0 = time.time()
    try:
        if LLM_MODE == "ollama":
            response_text = call_ollama(messages)
        else:
            response_text = call_openclaw(last_user_message)
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        response_text = "D\u00e9sol\u00e9, j'ai eu un petit probl\u00e8me. R\u00e9essaie!"

    log.info(f"Response in {time.time() - t0:.1f}s: '{response_text[:100]}...'")

    # Check if client wants streaming
    stream = body.get("stream", False)
    if stream:
        return StreamingResponse(
            make_streaming_chunks(response_text),
            media_type="text/event-stream",
        )
    return JSONResponse(content=make_openai_response(response_text))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
