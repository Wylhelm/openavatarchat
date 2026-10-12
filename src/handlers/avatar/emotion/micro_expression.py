from dataclasses import dataclass
from typing import Dict

import numpy as np

# Brow lift on stressed syllables, detected from the lip-sync jaw opening
EMPHASIS_JAW_THRESHOLD = 0.3
EMPHASIS_JAW_RANGE = 0.4
EMPHASIS_SMOOTHING = 0.25
EMPHASIS_MORPHS = {"browInnerUp": 0.15, "browOuterUpLeft": 0.1, "browOuterUpRight": 0.08}

# Slow, slightly asymmetric drift so the face never looks frozen: (amplitude, frequency Hz, phase)
DRIFT_MORPHS = {
    "browOuterUpLeft": (0.05, 0.23, 0.0),
    "browOuterUpRight": (0.04, 0.31, 1.3),
    "mouthSmileLeft": (0.03, 0.17, 2.1),
    "mouthSmileRight": (0.025, 0.13, 0.7),
    "eyeSquintLeft": (0.03, 0.19, 0.4),
    "eyeSquintRight": (0.03, 0.21, 1.9),
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
    for name, (amplitude, frequency, phase) in DRIFT_MORPHS.items():
        index = channel_index.get(name)
        if index is not None:
            offsets[:, index] += amplitude * 0.5 * (1.0 + np.sin(2.0 * np.pi * frequency * t + phase))

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
