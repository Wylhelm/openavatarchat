import os
import sys
import json
import subprocess
import time
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()

AVATAR_BRIDGE_API_KEY = os.environ.get("AVATAR_BRIDGE_API_KEY")
if not AVATAR_BRIDGE_API_KEY:
    print("FATAL: AVATAR_BRIDGE_API_KEY is not set. Refusing to start.", file=sys.stderr)
    sys.exit(1)

AVATAR_SAFE_MODE = os.environ.get("AVATAR_SAFE_MODE", "true").lower() in ("true", "1", "yes")
OPENCLAW_SESSION_ID = os.environ.get("OPENCLAW_TELEGRAM_ID") or os.environ.get("OPENCLAW_SESSION_ID", "main")
PORT = int(os.environ.get("PORT", "9000"))

# SSH mode: run openclaw on remote VM via SSH instead of locally
SSH_HOST = os.environ.get("OPENCLAW_SSH_HOST", "")
SSH_USER = os.environ.get("OPENCLAW_SSH_USER", "guillaume")
SSH_KEY = os.environ.get("OPENCLAW_SSH_KEY", "")

BLOCKED_PATTERNS = [
    "rm ",
    "rm -",
    "mv ",
    "docker rm",
    "docker stop",
    "sudo ",
    "pip install",
    "apt install",
    "ssh ",
    "scp ",
    "~/.openclaw/secrets",
    "gog gmail send",
    "gog calendar create",
    "gog gmail delete",
    "> /",
    ">> /",
    "| bash",
    "| sh",
    "curl -x post",
    "curl --data",
    "wget -o",
    "eval ",
    "exec(",
]

SAFE_MODE_BLOCKED_RESPONSE = "Je suis en mode vocal sécurisé. Pour cette action, confirme-moi ça sur Telegram."

app = FastAPI(title="Avatar Bridge", version="1.0.0")
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
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "safe_mode": AVATAR_SAFE_MODE}


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": "gbot",
                "object": "model",
                "created": 0,
                "owned_by": "avatar-bridge",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    body = await request.json()
    messages = body.get("messages", [])

    # Extract last user message
    last_user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_message = msg.get("content", "")
            break

    if last_user_message is None:
        raise HTTPException(status_code=400, detail="No user message found in messages array")

    # Safe mode check
    if AVATAR_SAFE_MODE and is_message_blocked(last_user_message):
        return JSONResponse(content=make_openai_response(SAFE_MODE_BLOCKED_RESPONSE))

    # Forward to OpenClaw (local or via SSH)
    try:
        if SSH_HOST:
            # Run openclaw on remote VM via SSH
            ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
            if SSH_KEY:
                ssh_cmd += ["-i", SSH_KEY]
            ssh_cmd += [
                f"{SSH_USER}@{SSH_HOST}",
                f"openclaw agent --to {OPENCLAW_SESSION_ID} --message {json.dumps(last_user_message)} --json"
            ]
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=120)
        else:
            result = subprocess.run(
                [
                    "openclaw",
                    "agent",
                    "--to", OPENCLAW_SESSION_ID,
                    "--message", last_user_message,
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="OpenClaw subprocess timed out")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="openclaw/ssh command not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subprocess error: {str(e)}")

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"OpenClaw returned error (exit {result.returncode}): {result.stderr.strip()}",
        )

    # Parse JSON response from OpenClaw
    try:
        openclaw_data = json.loads(result.stdout)
        # Try to extract text content from the response
        response_text = (
            openclaw_data.get("response")
            or openclaw_data.get("text")
            or openclaw_data.get("content")
            or openclaw_data.get("message")
            or str(openclaw_data)
        )
    except json.JSONDecodeError:
        # If not valid JSON, use stdout as plain text
        response_text = result.stdout.strip()

    return JSONResponse(content=make_openai_response(response_text))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
