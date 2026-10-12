# 🎭 Phase 1 — Réactions faciales émotionnelles (vision + ton de voix)

## Objectif

L'avatar affiche une **émotion sur son visage** en plus du mouvement de bouche :
- **Source 1 (vision)** : il déduit une émotion de ce qu'il **voit** (la scène caméra)
- **Source 2 (texte)** : il déduit une émotion de ce qu'il **dit** (le ton de sa réponse)

Le tout se **fusionne** avec les 52 morphs ARKit déjà pilotés par l'audio.

---

## Rappel du flux actuel

```
Caméra ──► [LLM handler] ──► texte ──► [TTS] ──► audio
                                                    │
                                          [LAM handler] ──► audio ──► 52 morphs ──► visage
```

Il faut faire transiter l'**émotion** du LLM handler jusqu'au LAM handler.

---

## Fichiers concernés (tous sur 10.0.0.175)

| Fichier | Action |
|---------|--------|
| `src/handlers/avatar/emotion/emotion_arkit.py` | **CRÉER** — table émotion→morphs + fusion |
| `src/handlers/llm/openai_compatible/llm_handler_openai_compatible.py` | **MODIFIER** — détecter/transmettre l'émotion |
| `src/handlers/avatar/lam/avatar_handler_lam_audio2expression.py` | **MODIFIER** — fusionner l'émotion dans les morphs |
| `config/chat_with_gbot.yaml` | **MODIFIER** — ajuster le prompt système |

---

## Étape 1 — Créer `emotion_arkit.py`

**Chemin :** `src/handlers/avatar/emotion/emotion_arkit.py`

Ce module charge les canaux depuis `arkit_face_channels.txt` (pas de hardcode) et expose la table + la fusion.

```python
import os
from typing import Dict, List

# Emotion -> intensités ARKit (0.0 à 1.0), par NOM de canal
EMOTION_MORPHS: Dict[str, Dict[str, float]] = {
    "surprise": {
        "browInnerUp": 0.7, "browOuterUpLeft": 0.4, "browOuterUpRight": 0.4,
        "eyeWideLeft": 0.6, "eyeWideRight": 0.6, "jawOpen": 0.25,
    },
    "joie": {
        "mouthSmileLeft": 0.7, "mouthSmileRight": 0.7,
        "cheekSquintLeft": 0.5, "cheekSquintRight": 0.5,
        "eyeSquintLeft": 0.3, "eyeSquintRight": 0.3,
    },
    "tristesse": {
        "mouthFrownLeft": 0.6, "mouthFrownRight": 0.6,
        "browInnerUp": 0.5, "eyeLookDownLeft": 0.3, "eyeLookDownRight": 0.3,
    },
    "colere": {
        "browDownLeft": 0.6, "browDownRight": 0.6,
        "mouthPressLeft": 0.4, "mouthPressRight": 0.4,
        "noseSneerLeft": 0.4, "noseSneerRight": 0.4,
    },
    "reflexion": {
        "eyeLookUpLeft": 0.5, "eyeLookUpRight": 0.5,
        "browDownLeft": 0.3, "browDownRight": 0.3,
    },
    "neutre": {},
}


def load_arkit_channels(channels_path: str) -> List[str]:
    """Charge l'ordre des 52 canaux depuis arkit_face_channels.txt."""
    with open(channels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_channel_index(channels: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(channels)}


def emotion_to_arkit(emotion: str, channel_index: Dict[str, int]) -> List[float]:
    """Convertit une émotion en vecteur [N] d'intensités ARKit (0.0 par défaut)."""
    vec = [0.0] * len(channel_index)
    for name, value in EMOTION_MORPHS.get(emotion, {}).items():
        if name in channel_index:
            vec[channel_index[name]] = value
    return vec


def blend_emotion(base_arkit, emotion_vec, weight: float = 0.6):
    """Mélange les morphs audio (base_arkit) avec l'émotion (additif, clampé [0,1]).

    base_arkit : np.ndarray [N] (mouvement de bouche du LAM)
    emotion_vec : list [N] (intensités émotion)
    weight : force de l'émotion (0..1)
    """
    import numpy as np
    result = base_arkit.copy()
    for i, v in enumerate(emotion_vec):
        if v != 0.0:
            result[i] = min(1.0, float(result[i]) + v * weight)
    return result
```

---

## Étape 2 — Modifier le prompt système (`chat_with_gbot.yaml`)

Sous `LLMOpenAICompatible`, remplace le `system_prompt` pour que le modèle **préfixe chaque réponse d'un tag émotionnel** :

```yaml
LLMOpenAICompatible:
  enabled: True
  module: llm/openai_compatible/llm_handler_openai_compatible
  model_name: "gbot"
  enable_video_input: True
  history_length: 12
  system_prompt: |
    Tu es G-Bot, l'assistant vocal de Guillaume.
    Réponds TOUJOURS en préfixant ta réponse par un tag d'émotion entre crochets,
    suivi du texte. Format exact : [EMOTION] texte
    EMOTION ∈ {neutre, joie, surprise, tristesse, colere, reflexion}
    Choisis l'émotion selon le ton de TA réponse et ce que tu vois à la caméra.
    Exemples :
    [joie] Bonjour Guillaume, ça me fait plaisir de te voir !
    [surprise] Oh, tu as changé de coiffure !
    [neutre] Il est 14h, tu as une réunion dans 30 minutes.
    Réponds en français, 2-3 phrases maximum.
```

---

## Étape 3 — Modifier `llm_handler_openai_compatible.py`

### 3a. Détecter l'émotion dans la réponse

À la fin du handler, quand la réponse complète est prête (`context.output_texts`), extraire le tag et le stocker, puis le transmettre en **meta** du DataBundle.

```python
import re

# Après la boucle de streaming, quand context.output_texts est complet :
emotion = "neutre"
m = re.match(r"^\s*\[(\w+)\]\s*", context.output_texts)
if m:
    emotion = m.group(1).lower()
    # Retirer le tag du texte envoyé au TTS
    context.output_texts = context.output_texts[m.end():].strip()
```

### 3b. Transmettre l'émotion en meta

Dans la boucle de streaming, sur **chaque** `output`, ajouter :

```python
output.add_meta("avatar_emotion", emotion)
```

Et sur le `end_output` final aussi :

```python
end_output.add_meta("avatar_emotion", emotion)
```

> ⚠️ **Attention streaming** : le tag est en début de réponse, mais le texte est émis chunk par chunk. Le plus simple est de **bufferiser la réponse complète**, extraire l'émotion, PUIS émettre les chunks nettoyés. Ou, à défaut, n'émettre l'émotion que sur le `end_output` final (le LAM la lira à ce moment-là).

---

## Étape 4 — Modifier `avatar_handler_lam_audio2expression.py`

### 4a. Charger les canaux + l'index (au `load()`)

```python
from handlers.avatar.emotion.emotion_arkit import (
    load_arkit_channels, build_channel_index, emotion_to_arkit, blend_emotion,
)

# Dans load(), après avoir chargé self.arkit_channels :
self.channel_index = build_channel_index(self.arkit_channels)
```

### 4b. Fusionner l'émotion dans le `handle()`

Avant `output.set_main_data(arkit_data...)`, récupérer l'émotion et la fusionner :

```python
# Récupérer l'émotion (transmise du LLM handler via meta)
emotion = inputs.data.get_meta("avatar_emotion") or "neutre"

# ... après avoir calculé arkit_data (tableau [1, N]) ...
emotion_vec = emotion_to_arkit(emotion, self.channel_index)
arkit_data = blend_emotion(arkit_data, emotion_vec, weight=0.6)

output.set_main_data(arkit_data.astype(np.float32))
```

---

## ⚠️ Les 3 points à déboguer (honnêteté)

1. **Propagation de la meta `avatar_emotion`** du LLM handler → TTS → LAM handler. C'est LE point incertain : vérifier que le `DataBundle` transporte bien la meta à travers ton pipeline. Si le TTS ne la propage pas, il faudra un autre canal (ex: un dictionnaire partagé dans le `SessionContext`).

2. **Alignement des 52 canaux** : l'ordre de `arkit_face_channels.txt` doit matcher l'index. Comme on charge depuis le fichier, c'est automatique — juste vérifier que le LAM handler charge bien les canaux dans le même ordre que son tableau de sortie.

3. **Parsing du tag en streaming** : extraire `[emotion]` proprement sans le laisser passer dans le TTS.

---

## 🔁 Alternative plus simple si le point 1 bloque

Si la propagation de meta est trop galère, **infère l'émotion directement dans le LAM handler** depuis `avatar_speech_text` (déjà dispo dans le code). Ça couvre la source "ce qu'il dit", mais **pas** la vision. Pour la vision→émotion, il faudra de toute façon un canal LLM→LAM.

---

## Ordre recommandé

1. **Étape 1** (module émotion) — indépendant, testable
2. **Étape 2+3** (prompt + détection) — teste que le tag est bien extrait
3. **Étape 4** (fusion morphs) — le visage réagit
4. **Déboguer** le point 1 si l'émotion n'arrive pas au LAM
