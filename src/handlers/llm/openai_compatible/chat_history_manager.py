from dataclasses import dataclass
import re
from typing import Literal, Optional
from loguru import logger


from engine_utils.media_utils import ImageUtils


@dataclass
class HistoryMessage:
    role: Optional[Literal['avatar', 'human']] = None
    content: str = ''
    timestamp: Optional[str] = None


name_dict = {
    "avatar": "assistant",
    "human": "user"
}


def filter_text(text):
    # Preserve French accented characters (é, è, ê, à, ç, etc.), English, numbers, Chinese, and punctuation
    # Use UNICODE word characters (\w) which includes accented letters, plus explicit French characters
    pattern = r"[^\w\s\u4e00-\u9fff,.\-~!?;:\'\"()àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸ，。！？]"  # Preserve French accents
    filtered_text = re.sub(pattern, "", text, flags=re.UNICODE)
    return filtered_text


def normalize_for_comparison(text: str) -> str:
    """Normalize text for similarity comparison"""
    if not text:
        return ""
    # Remove extra whitespace, convert to lowercase, remove punctuation
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    normalized = re.sub(r'[^\w\s\u4e00-\u9fff]', '', normalized)
    return normalized


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts (0.0 to 1.0)"""
    norm1 = normalize_for_comparison(text1)
    norm2 = normalize_for_comparison(text2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Simple word overlap similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


class ChatHistory:
    def __init__(self, history_length):
        self.max_history_length = history_length
        self.message_history = []

    def is_duplicate_response(self, new_content: str, similarity_threshold: float = 0.85) -> bool:
        """Check if the new response is too similar to recent avatar responses"""
        if not new_content or len(new_content.strip()) < 10:
            return False
        
        # Check against the last few avatar responses
        avatar_responses = [
            msg.content for msg in self.message_history 
            if msg.role == 'avatar' and msg.content
        ]
        
        # Only check last 2-3 responses to avoid false positives
        for recent_response in avatar_responses[-3:]:
            similarity = calculate_similarity(new_content, recent_response)
            if similarity >= similarity_threshold:
                return True
        
        return False

    def add_message(self, message: HistoryMessage, check_duplicates: bool = True):
        history = self.message_history
        
        # Check for duplicate avatar responses before adding
        if check_duplicates and message.role == 'avatar':
            if self.is_duplicate_response(message.content):
                # Don't add duplicate response, but still maintain history length
                # This prevents reinforcing looping behavior
                logger.warning(f"Duplicate avatar response detected and skipped to prevent looping: {message.content[:100]}...")
                return
        
        history.append(message)
        # thread safe
        while len(history) >= self.max_history_length:
            history.pop(0)

    def generate_next_messages(self, chat_text, images):
        def history_to_message(history: HistoryMessage):
            return {
                "role": name_dict[history.role],
                "content": filter_text(history.content),
            }
        history = self.message_history
        messages = list(map(history_to_message, history))
        if images and len(images) > 0:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": filter_text(chat_text),
                    },
                ] + (list(map(lambda x: {"type": "image_url", "image_url": {"url": ImageUtils.format_image(x)}}, images)))
            })
        else: 
            messages.append({
                "role": "user",
                "content": filter_text(chat_text),
            })
        return messages        
    

  