import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np

NEUTRAL_EMOTION = "neutre"

# Emotion -> ARKit channel intensities (0.0 to 1.0), by channel name
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
    # Nuanced, often asymmetric expressions
    "amuse": {
        "mouthSmileLeft": 0.55, "mouthSmileRight": 0.2,
        "cheekSquintLeft": 0.35, "browOuterUpLeft": 0.2,
    },
    "doux": {
        "mouthSmileLeft": 0.25, "mouthSmileRight": 0.25,
        "eyeSquintLeft": 0.3, "eyeSquintRight": 0.3, "browInnerUp": 0.2,
    },
    "curieux": {
        "browOuterUpLeft": 0.6, "browInnerUp": 0.3, "eyeWideLeft": 0.25,
    },
    "sceptique": {
        "browOuterUpRight": 0.5, "browDownLeft": 0.35,
        "mouthPressLeft": 0.3, "mouthRight": 0.15,
    },
    "inquiet": {
        "browInnerUp": 0.6, "eyeWideLeft": 0.2, "eyeWideRight": 0.2,
        "mouthStretchLeft": 0.2, "mouthStretchRight": 0.2,
    },
    NEUTRAL_EMOTION: {},
}

MAX_EMOTION_INTENSITY = 1.0

# Mouth and jaw channels are driven by lip-sync; emotion is damped there
LIP_SYNC_CHANNEL_PREFIXES = ("mouth", "jaw")
LIP_SYNC_DAMPING = 0.5


def parse_emotion(raw) -> Optional[Tuple[str, float]]:
    """Parses "joie" or "joie:0.4" into (accent-free name, intensity), or None if the name is unknown."""
    if not raw:
        return None
    text = unicodedata.normalize("NFKD", str(raw).strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    name, _, intensity_text = text.partition(":")
    name = name.strip()
    if name not in EMOTION_MORPHS:
        return None
    intensity = 1.0
    if intensity_text.strip():
        try:
            intensity = float(intensity_text.strip().replace(",", "."))
        except ValueError:
            return None
    return name, min(max(intensity, 0.0), MAX_EMOTION_INTENSITY)


def canonical_emotion(raw) -> Optional[str]:
    """Canonical "name" or "name:0.40" string, or None if it is not a known emotion."""
    parsed = parse_emotion(raw)
    if parsed is None:
        return None
    name, intensity = parsed
    return name if intensity >= MAX_EMOTION_INTENSITY else f"{name}:{intensity:.2f}"


def normalize_emotion(raw) -> str:
    return canonical_emotion(raw) or NEUTRAL_EMOTION


def load_arkit_channels(channels_path: str) -> List[str]:
    with open(channels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_channel_index(channels: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(channels)}


def emotion_to_arkit(emotion: str, channel_index: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(channel_index), dtype=np.float32)
    emotion_name, intensity = parse_emotion(emotion) or (NEUTRAL_EMOTION, 0.0)
    for name, value in EMOTION_MORPHS[emotion_name].items():
        index = channel_index.get(name)
        if index is None:
            continue
        if name.startswith(LIP_SYNC_CHANNEL_PREFIXES):
            value *= LIP_SYNC_DAMPING
        vec[index] = value * intensity
    return vec


def ramp_levels(start_level: float, frame_count: int, ramp_frames: float, fade_out: bool = False):
    """Per-frame emotion levels ramping up from start_level, optionally fading to 0 at the end.

    Returns (levels of shape [frame_count, 1], level to carry into the next call).
    """
    if frame_count <= 0:
        return np.zeros((0, 1), dtype=np.float32), start_level
    step = 1.0 / max(ramp_frames, 1.0)
    levels = np.clip(start_level + step * np.arange(1, frame_count + 1), 0.0, 1.0).astype(np.float32)
    end_level = float(levels[-1])
    if fade_out:
        fade_frames = min(frame_count, max(int(round(ramp_frames)), 1))
        levels[-fade_frames:] *= np.linspace(1.0, 0.0, fade_frames, dtype=np.float32)
        end_level = 0.0
    return levels[:, np.newaxis], end_level


def blend_emotion(base_arkit: np.ndarray, emotion_vec: np.ndarray, weight=0.6) -> np.ndarray:
    """Additive blend of emotion onto lip-sync morphs, clamped to [0, 1].

    base_arkit has the channel axis last ([N], [1, N] or [T, N]).
    weight is a scalar or a per-frame array of shape [T, 1].
    """
    weight = np.asarray(weight, dtype=np.float32)
    if not np.any(weight > 0.0) or not np.any(emotion_vec):
        return base_arkit
    blended = base_arkit + weight * emotion_vec
    return np.clip(blended, 0.0, 1.0).astype(base_arkit.dtype, copy=False)
