import re
from typing import Optional, Tuple

from handlers.avatar.emotion.emotion_arkit import NEUTRAL_EMOTION, canonical_emotion

MAX_TAG_BUFFER_CHARS = 26
_LEADING_TAG_RE = re.compile(r"^\s*\[([^\[\]]{1,20})\]\s*")
_ANY_TAG_RE = re.compile(r"\[([^\[\]]{1,20})\]\s*")


def strip_emotion_tags(text: str) -> str:
    """Remove every [emotion] tag naming a known emotion, leaving other bracketed text intact."""
    if not text:
        return text
    return _ANY_TAG_RE.sub(lambda m: "" if canonical_emotion(m.group(1)) else m.group(0), text).strip()


class EmotionTagStreamParser:
    """Strips a leading [emotion] tag from a streamed LLM response.

    Leading text is held back only while it could still be a tag, so TTS never receives the tag.
    """

    def __init__(self):
        self._buffer = ""
        self._resolved = False
        self._skip_leading_space = False
        self.emotion: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self._resolved

    def feed(self, chunk: str) -> Tuple[str, bool]:
        """Returns (text to emit, whether the emotion was resolved by this chunk)."""
        if self._resolved:
            return self._emit(chunk), False
        self._buffer += chunk
        stripped = self._buffer.lstrip()
        if not stripped:
            return "", False
        if not stripped.startswith("["):
            return self._resolve(NEUTRAL_EMOTION, self._buffer), True
        match = _LEADING_TAG_RE.match(self._buffer)
        if match:
            emotion = canonical_emotion(match.group(1))
            if emotion:
                self._skip_leading_space = True
                return self._resolve(emotion, self._buffer[match.end():]), True
            return self._resolve(NEUTRAL_EMOTION, self._buffer), True
        if "]" in stripped or len(self._buffer) > MAX_TAG_BUFFER_CHARS:
            return self._resolve(NEUTRAL_EMOTION, self._buffer), True
        return "", False

    def flush(self) -> Tuple[str, bool]:
        """Releases any held-back text at the end of the stream."""
        if self._resolved:
            return "", False
        return self._resolve(NEUTRAL_EMOTION, self._buffer), True

    def _resolve(self, emotion: str, text: str) -> str:
        self._resolved = True
        self._buffer = ""
        self.emotion = emotion
        return self._emit(text)

    def _emit(self, text: str) -> str:
        if self._skip_leading_space:
            text = text.lstrip()
            if text:
                self._skip_leading_space = False
        return text
