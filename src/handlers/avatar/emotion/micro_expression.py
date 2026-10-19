from dataclasses import dataclass
from typing import Dict

import numpy as np

# Brow lift & cheek squint on stressed syllables, detected from the lip-sync jaw opening
EMPHASIS_JAW_THRESHOLD = 0.28
EMPHASIS_JAW_RANGE = 0.4
EMPHASIS_SMOOTHING = 0.25
EMPHASIS_MORPHS = {
    "browInnerUp": 0.15,
    "browOuterUpLeft": 0.11,
    "browOuterUpRight": 0.085,
    "cheekSquintLeft": 0.06,
    "cheekSquintRight": 0.055,
}

# Multi-harmonic, slightly asymmetric drift so the face never looks frozen:
# (amplitude, primary_freq_hz, phase, secondary_freq_hz, secondary_phase)
DRIFT_MORPHS = {
    "browOuterUpLeft": (0.045, 0.23, 0.0, 0.51, 1.1),
    "browOuterUpRight": (0.038, 0.31, 1.3, 0.47, 2.4),
    "browInnerUp": (0.03, 0.19, 0.8, 0.43, 0.2),
    "mouthSmileLeft": (0.03, 0.17, 2.1, 0.37, 0.5),
    "mouthSmileRight": (0.026, 0.13, 0.7, 0.41, 1.8),
    "cheekSquintLeft": (0.025, 0.17, 2.0, 0.39, 0.9),
    "cheekSquintRight": (0.022, 0.15, 0.9, 0.35, 1.5),
    "eyeSquintLeft": (0.028, 0.19, 0.4, 0.43, 2.2),
    "eyeSquintRight": (0.028, 0.21, 1.9, 0.37, 0.6),
}


@dataclass
class MicroExpressionState:
    frame: int = 0
    emphasis: float = 0.0


def micro_expression_offsets(frames: np.ndarray, channel_index: Dict[str, int], state: MicroExpressionState,
                             frame_rate: float) -> np.ndarray:
    """Per-frame additive offsets [T, N] for the lip-sync frames [T, N]; advances state."""
    frame_count, channel_count = frames.shape
    offsets = np.zeros((frame_count, channel_count), dtype=np.float32)
    if frame_count == 0:
        return offsets

    t = (state.frame + np.arange(frame_count, dtype=np.float32)) / frame_rate
    for name, (amplitude, freq1, phase1, freq2, phase2) in DRIFT_MORPHS.items():
        index = channel_index.get(name)
        if index is not None:
            wave1 = 0.5 * (1.0 + np.sin(2.0 * np.pi * freq1 * t + phase1))
            wave2 = 0.5 * (1.0 + np.sin(2.0 * np.pi * freq2 * t + phase2))
            offsets[:, index] += amplitude * (0.65 * wave1 + 0.35 * wave2)

    jaw_index = channel_index.get("jawOpen")
    if jaw_index is not None:
        target = np.clip((frames[:, jaw_index] - EMPHASIS_JAW_THRESHOLD) / EMPHASIS_JAW_RANGE, 0.0, 1.0)
        emphasis = np.empty(frame_count, dtype=np.float32)
        level = state.emphasis
        for i, value in enumerate(target):
            level += EMPHASIS_SMOOTHING * (value - level)
            emphasis[i] = level
        state.emphasis = float(level)
        for name, amplitude in EMPHASIS_MORPHS.items():
            index = channel_index.get(name)
            if index is not None:
                offsets[:, index] += amplitude * emphasis

    state.frame += frame_count
    return offsets
