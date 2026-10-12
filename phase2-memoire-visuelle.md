# 🧠 Phase 2 — Mémoire visuelle (scènes & objets)

## Objectif

L'avatar **se souvient de ce qu'il voit** : il décrit la scène caméra, la stocke horodatée, et peut la rappeler ensuite ("tu te souviens de... ?").

- **Portée** : scènes / objets (description générale), PAS de reconnaissance de visages pour l'instant.
- **Persistance** : un fichier JSON local, horodaté.

---

## Rappel du flux actuel (sur 10.0.0.175)

```
Caméra ──► [LLM handler] ──► texte ──► [TTS] ──► audio ──► [LAM] ──► visage
                │
                └── (l'image est actuellement JETÉE après usage)
```

L'image caméra arrive bien dans le handler LLM (`CAMERA_VIDEO`), mais elle est **jetée** après la réponse (`context.current_image = None`). Il faut la **décrire** et la **stocker** au lieu de la perdre.

---

## Fichiers concernés (tous sur 10.0.0.175)

| Fichier | Action |
|---------|--------|
| `src/handlers/llm/openai_compatible/vision_memory.py` | **CRÉER** — journalisation + rappel |
| `src/handlers/llm/openai_compatible/llm_handler_openai_compatible.py` | **MODIFIER** — décrire + stocker + injecter le rappel |
| `config/chat_with_gbot.yaml` | **MODIFIER** — ajuster le prompt si besoin |

---

## Étape 1 — Créer `vision_memory.py`

**Chemin :** `src/handlers/llm/openai_compatible/vision_memory.py`

Module de persistance : écrit dans un fichier JSONL horodaté, et lit les dernières entrées pour le rappel.

```python
import json
import os
import time
from typing import List, Dict, Optional

# Chemin du journal de mémoire visuelle (ajuster selon l'utilisateur de l'avatar)
MEMORY_PATH = os.path.expanduser("~/.openclaw/avatar-memory/vision_log.jsonl")


def log_vision(description: str, emotion: str = "neutre") -> None:
    """Ajoute une entrée de mémoire visuelle horodatée."""
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "description": description,
        "emotion": emotion,
    }
    with open(MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_recent_vision(limit: int = 10) -> List[Dict]:
    """Retourne les dernières entrées de mémoire visuelle (la plus récente en dernier)."""
    if not os.path.exists(MEMORY_PATH):
        return []
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[-limit:]


def format_vision_context(entries: List[Dict]) -> str:
    """Formate les entrées en bloc texte injectable dans le prompt."""
    if not entries:
        return ""
    lines = [f"- {e.get('ts', '?')}: {e.get('description', '')}" for e in entries]
    return "\n".join(lines)
```

---

## Étape 2 — Modifier `llm_handler_openai_compatible.py`

### 2a. Décrire et stocker la scène

Quand une image caméra arrive, au lieu de la jeter, on **décrit** la scène et on la **stocke**.

Deux approches possibles — **Approche A recommandée** (appel LLM dédié, fiable et isolé) :

```python
# Dans le handle(), quand inputs.type == CAMERA_VIDEO et enable_video_input :
context.current_image = inputs.data.get_main_data()

# NOUVEAU : décrire la scène pour la mémoire
# (appel LLM séparé, ne perturbe pas la réponse principale)
try:
    from handlers.llm.openai_compatible.vision_memory import log_vision
    # Petit appel de description (le client OpenAI est déjà dispo : context.client)
    desc_resp = context.client.chat.completions.create(
        model=context.model_name,
        messages=[
            {"role": "system", "content": "Décris la scène en une phrase concise, en français. Ce que tu vois, les objets, les personnes (sans nom), l'ambiance."},
            {"role": "user", "content": [
                {"type": "text", "text": "Décris ce que tu vois."},
                {"type": "image_url", "image_url": {"url": context.current_image}},
            ]},
        ],
        max_tokens=60,
    )
    description = desc_resp.choices[0].message.content.strip()
    log_vision(description)
except Exception as e:
    logger.warning(f"Vision memory log failed: {e}")
```

> ⚠️ **Latence** : cet appel dédié ajoute un aller-retour LLM. Si tu veux l'éviter, voir l'**Approche B** plus bas.

### 2b. Injecter le rappel dans le prompt

Quand l'avatar reçoit une question de type "tu te souviens...", injecter les dernières descriptions dans le `system_prompt`.

```python
# Dans create_context() ou handle(), avant l'appel LLM principal :
from handlers.llm.openai_compatible.vision_memory import load_recent_vision, format_vision_context

recent = load_recent_vision(limit=5)
if recent:
    context_block = format_vision_context(recent)
    context.system_prompt["content"] += (
        "\n\nMémoire visuelle récente (tu peux t'y référer si on te le demande):\n"
        + context_block
    )
```

---

## Étape 3 — Modifier `chat_with_gbot.yaml` (optionnel)

Ajouter au `system_prompt` une règle indiquant que l'avatar **se souvient** de ce qu'il a vu :

```yaml
system_prompt: |
  Tu es G-Bot, l'assistant vocal de Guillaume.
  Tu as une mémoire visuelle : tu te souviens des scènes que tu as observées
  à la caméra (elles sont listées dans ton contexte).
  Si Guillaume te demande "tu te souviens de...", réfère-toi à cette mémoire.
  [EMOTION] texte   # (si la Phase 1 est déjà en place)
  Réponds en français, 2-3 phrases maximum.
```

---

## ⚠️ Points à déboguer (honnêteté)

1. **Latence de l'appel dédié** (Approche A) : un second appel LLM allonge le temps de réponse. Si c'est trop lent, passer à l'Approche B.

2. **Format de `context.current_image`** : vérifier que c'est bien un `data:image/...;base64` utilisable directement dans `image_url`. Le handler le construit via `ImageUtils.format_image`, donc normalement oui.

3. **Chemin du fichier mémoire** : `~/.openclaw/avatar-memory/` doit être accessible en écriture par le process avatar sur 10.0.0.175. Ajuster `MEMORY_PATH` si besoin.

---

## 🔁 Approche B (alternative, sans second appel LLM)

Au lieu d'un appel dédié, demander au modèle, **dans sa réponse principale**, d'inclure une description de scène. Ex: le prompt lui demande de répondre avec un bloc `[MEMOIRE] description`. On parse ce bloc à la fin et on le stocke.

- **Avantage** : pas de latence supplémentaire.
- **Inconvénient** : parsing fragile en streaming, et mélange le texte de réponse avec la description.

Je recommande **A** pour démarrer (plus simple à raisonner), puis optimiser vers **B** si la latence gêne.

---

## Ordre recommandé

1. **Étape 1** (module `vision_memory.py`) — indépendant, testable
2. **Étape 2a** (décrire + stocker) — teste que le fichier se remplit
3. **Étape 2b** (injecter le rappel) — teste "tu te souviens de... ?"
4. **Étape 3** (prompt) — peaufiner le comportement

---

## 🔗 Lien avec la Phase 1

Les deux phases partagent le même handler LLM (`llm_handler_openai_compatible.py`). Si tu fais les deux, intègre-les proprement dans le même `handle()` : l'émotion (Phase 1) et la description mémoire (Phase 2) peuvent être calculées au même endroit, sur la même image.
