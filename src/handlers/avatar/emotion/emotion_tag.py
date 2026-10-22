import re
from typing import Optional, Tuple

from handlers.avatar.emotion.emotion_arkit import NEUTRAL_EMOTION, canonical_emotion

MAX_TAG_BUFFER_CHARS = 26
_LEADING_TAG_RE = re.compile(r"^\s*\[([^\[\]]{1,20})\]\s*")
_ANY_TAG_RE = re.compile(r"\[([^\[\]]{1,20})\][ \t]*")


def strip_emotion_tags(text: str) -> str:
    """Remove every [emotion] tag naming a known emotion, leaving other bracketed text intact."""
    if not text:
        return text
    return _ANY_TAG_RE.sub(lambda m: "" if canonical_emotion(m.group(1)) else m.group(0), text).strip()


class EmotionTagStreamParser:
    """Strips leading and inline [emotion] tags from a streamed LLM response.

    Bracketed prefixes are held back only while they could still be an emotion tag,
    so TTS and chat subtitles never receive raw [emotion] tags even in multi-paragraph responses.
    """

    def __init__(self):
        self._buffer = ""
        self._inline_buffer = ""
        self._resolved = False
        self._skip_leading_space = False
        self.emotion: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self._resolved

    def feed(self, chunk: str) -> Tuple[str, bool]:
        """Returns (text to emit, whether the emotion was resolved/updated by this chunk)."""
        if self._resolved:
            return self._scan_inline(chunk, initial_resolved=False)
        self._buffer += chunk
        stripped = self._buffer.lstrip()
        if not stripped:
            return "", False
        if not stripped.startswith("["):
            return self._resolve(NEUTRAL_EMOTION, self._buffer)
        # If a tool call restarted generation mid-tag (e.g. "[sur[sceptique]"),
        # discard the unclosed prefix before the latest "[".
        if "[" in stripped[1:]:
            first_close = stripped.find("]")
            second_open = stripped.find("[", 1)
            if first_close == -1 or second_open < first_close:
                last_open_before_close = stripped.rfind("[", 0, first_close if first_close != -1 else len(stripped))
                self._buffer = stripped[last_open_before_close:]
                stripped = self._buffer
        match = _LEADING_TAG_RE.match(self._buffer)
        if match:
            emotion = canonical_emotion(match.group(1))
            if emotion:
                self._skip_leading_space = True
                return self._resolve(emotion, self._buffer[match.end():])
            return self._resolve(NEUTRAL_EMOTION, self._buffer)
        if "]" in stripped or len(self._buffer) > MAX_TAG_BUFFER_CHARS:
            return self._resolve(NEUTRAL_EMOTION, self._buffer)
        return "", False

    def flush(self) -> Tuple[str, bool]:
        """Releases any held-back text at the end of the stream."""
        if not self._resolved:
            text, resolved = self._resolve(NEUTRAL_EMOTION, self._buffer)
            if self._inline_buffer:
                tail = self._inline_buffer
                self._inline_buffer = ""
                text += self._emit(tail)
            return text, resolved
        if self._inline_buffer:
            tail = self._inline_buffer
            self._inline_buffer = ""
            emotion_updated = False

            def _replace(m):
                nonlocal emotion_updated
                em = canonical_emotion(m.group(1))
                if em:
                    self.emotion = em
                    emotion_updated = True
                    return ""
                return m.group(0)

            cleaned = _ANY_TAG_RE.sub(_replace, tail)
            return self._emit(cleaned), emotion_updated
        return "", False

    def _resolve(self, emotion: str, text: str) -> Tuple[str, bool]:
        self._resolved = True
        self._buffer = ""
        self.emotion = emotion
        return self._scan_inline(text, initial_resolved=True)

    def _scan_inline(self, text: str, initial_resolved: bool) -> Tuple[str, bool]:
        buf = self._inline_buffer + text
        self._inline_buffer = ""
        emotion_updated = initial_resolved

        def _replace(m):
            nonlocal emotion_updated
            em = canonical_emotion(m.group(1))
            if em:
                self.emotion = em
                emotion_updated = True
                return ""
            return m.group(0)

        buf = _ANY_TAG_RE.sub(_replace, buf)

        last_open = buf.rfind("[")
        if last_open != -1:
            last_close = buf.rfind("]")
            if last_close < last_open:
                tail_len = len(buf) - last_open
                if tail_len <= MAX_TAG_BUFFER_CHARS:
                    self._inline_buffer = buf[last_open:]
                    buf = buf[:last_open]

        return self._emit(buf), emotion_updated

    def _emit(self, text: str) -> str:
        if self._skip_leading_space:
            text = text.lstrip()
            if text:
                self._skip_leading_space = False
        return text
