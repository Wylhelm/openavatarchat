

import os
import re
from typing import Dict, Optional, cast
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
from openai import APIStatusError, OpenAI
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from handlers.llm.openai_compatible.chat_history_manager import ChatHistory, HistoryMessage


# Removed complex filtering - let the LLM and TTS work naturally


class LLMConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="qwen-plus")
    system_prompt: str = Field(default="请你扮演一个 AI 助手，用简短的对话来回答用户的问题，并在对话内容中加入合适的标点符号，不需要加入标点符号相关的内容")
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY"))
    api_url: str = Field(default=None)
    enable_video_input: bool = Field(default=False)
    history_length: int = Field(default=20)
    temperature: float = Field(default=0.7)  # Lower temperature = more focused, less creative
    max_tokens: int = Field(default=150)  # Limit response length


class LLMContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.model_name = None
        self.system_prompt = None
        self.api_key = None
        self.api_url = None
        self.client = None
        self.input_texts = ""
        self.output_texts = ""
        self.current_image = None
        self.history = None
        self.enable_video_input = False


class HandlerLLM(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=LLMConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_text_entry("avatar_text"))
        inputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
            ),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                type=ChatDataType.CAMERA_VIDEO,
            ),
        }
        outputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, LLMConfig):
            if handler_config.api_key is None or len(handler_config.api_key) == 0:
                error_message = 'api_key is required in config/xxx.yaml, when use handler_llm'
                logger.error(error_message)
                raise ValueError(error_message)

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, LLMConfig):
            handler_config = LLMConfig()
        context = LLMContext(session_context.session_info.session_id)
        context.config = handler_config  # Store config for temperature/max_tokens access
        context.model_name = handler_config.model_name
        context.system_prompt = {'role': 'system', 'content': handler_config.system_prompt}
        context.api_key = handler_config.api_key
        context.api_url = handler_config.api_url
        context.enable_video_input = handler_config.enable_video_input
        context.history = ChatHistory(history_length=handler_config.history_length)
        context.client = OpenAI(
            api_key=context.api_key,
            base_url=context.api_url,
            timeout=120.0,
        )
        return context
    
    def start_context(self, session_context, handler_context):
        pass

    def _remove_duplicate_endings(self, text):
        """Remove duplicate sentences, phrases, and words (aggressive cleanup for LLM looping)"""
        if not text or len(text.strip()) < 3:
            return text
        
        # First: Split into sentences and remove duplicate sentences
        sentences = re.split(r'([.!?。！？])', text)
        cleaned_sentences = []
        seen_sentences = set()
        
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences):
                sentence = (sentences[i] + sentences[i+1]).strip()
                i += 2
            else:
                sentence = sentences[i].strip()
                i += 1
            
            if not sentence:
                continue
            
            # Normalize sentence for comparison (lowercase, remove extra spaces)
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            
            # Skip if we've seen this exact sentence before
            if normalized in seen_sentences:
                continue
            
            # Also check if this sentence is a substring of a previous one (or vice versa)
            is_duplicate = False
            for seen in seen_sentences:
                if len(normalized) > 10 and len(seen) > 10:  # Only check longer sentences
                    if normalized in seen or seen in normalized:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                seen_sentences.add(normalized)
                cleaned_sentences.append(sentence)
        
        if not cleaned_sentences:
            # If all sentences were duplicates, keep just the first unique one
            first_sentence = re.split(r'([.!?。！？])', text)[:2]
            return ''.join(first_sentence) if len(first_sentence) >= 2 else text.strip()[:100]
        
        result = ' '.join(cleaned_sentences)
        
        # Second: Remove duplicate words within the result
        words = result.split()
        if len(words) > 1:
            cleaned_words = []
            for i, word in enumerate(words):
                if i == 0 or word != words[i-1]:
                    cleaned_words.append(word)
            
            # Check for repeated phrases (3+ word sequences)
            if len(cleaned_words) >= 6:
                # Check last 3 words against previous 3 words
                last_three = ' '.join(cleaned_words[-3:])
                prev_three = ' '.join(cleaned_words[-6:-3])
                if last_three == prev_three:
                    cleaned_words = cleaned_words[:-3]
            
            # Check for single word repetition at the end (e.g., "word word word word")
            if len(cleaned_words) >= 2:
                last_word = cleaned_words[-1]
                # Count how many times the last word appears consecutively at the end
                count = 0
                for i in range(len(cleaned_words) - 1, -1, -1):
                    if cleaned_words[i] == last_word:
                        count += 1
                    else:
                        break
                # If the same word appears 2+ times at the end, keep only one
                if count >= 2:
                    cleaned_words = cleaned_words[:-(count-1)]
            
            result = ' '.join(cleaned_words)
        
        # Preserve trailing punctuation
        if text.strip() and text.strip()[-1] in '.!?。！？':
            if not result.endswith(text.strip()[-1]):
                result = result.rstrip('.!?。！？') + text.strip()[-1]
        
        return result.strip()

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        context = cast(LLMContext, context)
        text = None
        if inputs.type == ChatDataType.CAMERA_VIDEO and context.enable_video_input:
            context.current_image = inputs.data.get_main_data()
            return
        elif inputs.type == ChatDataType.HUMAN_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if (speech_id is None):
            speech_id = context.session_id

        if text is not None:
            context.input_texts += text

        text_end = inputs.data.get_meta("human_text_end", False)
        if not text_end:
            return

        chat_text = context.input_texts
        chat_text = re.sub(r"<\|.*?\|>", "", chat_text)
        chat_text = chat_text.strip()
        
        if len(chat_text) < 1:
            return
        
        logger.info(f'LLM input: {chat_text}')
        current_content = context.history.generate_next_messages(chat_text, 
                                                                 [context.current_image] if context.current_image is not None else [])
        try:
            # Get temperature and max_tokens from config if available
            temperature = getattr(context.config, 'temperature', 0.7)
            max_tokens = getattr(context.config, 'max_tokens', 200)
            completion = context.client.chat.completions.create(
                model=context.model_name,
                messages=[
                    context.system_prompt,
                ] + current_content,
                stream=True,
                temperature=temperature,  # Lower temperature = more focused, less creative
                max_tokens=max_tokens,  # Limit response length
                stream_options={"include_usage": True}
            )
            context.current_image = None
            context.input_texts = ''
            context.output_texts = ''
            
            for chunk in completion:
                if (chunk and chunk.choices and chunk.choices[0] and chunk.choices[0].delta.content):
                    output_text = chunk.choices[0].delta.content
                    # Ensure UTF-8 encoding is preserved
                    if isinstance(output_text, bytes):
                        output_text = output_text.decode('utf-8')
                    
                    # Simple check for obvious infinite loops (same word repeated many times)
                    # Only stop if we see the exact same word/token repeated 5+ times in a row
                    if len(output_text.strip()) > 0:
                        words = output_text.strip().split()
                        if len(words) >= 5:
                            # Check if last 5 words are all the same
                            last_5_words = words[-5:]
                            if len(set(last_5_words)) == 1:  # All words are identical
                                logger.warning(f"Stopping LLM stream due to word repetition: '{last_5_words[0]}'")
                                break
                    
                    context.output_texts += output_text
                    # Log if we see "tre" which might indicate "être" was corrupted
                    if 'tre' in output_text.lower() and 'être' not in context.output_texts.lower():
                        logger.warning(f"Possible corruption: 'tre' detected in chunk: '{output_text}'")
                    logger.info(output_text)
                    output = DataBundle(output_definition)
                    output.set_main_data(output_text)
                    output.add_meta("avatar_text_end", False)
                    output.add_meta("speech_id", speech_id)
                    yield output
            
            # Post-process the complete response to ensure proper French punctuation and accents
            if context.output_texts:
                # Fix common French words that might be missing accents
                french_fixes = {
                    r'\betre\b': 'être',
                    r'\bEtre\b': 'Être',
                    r'\bETRE\b': 'ÊTRE',
                    r'\bfatigue\b': 'fatigué',
                    r'\bFatigue\b': 'Fatigué',
                    r'\bfrancais\b': 'français',
                    r'\bFrancais\b': 'Français',
                    r'\bFRANCAIS\b': 'FRANÇAIS',
                    r'\bdeja\b': 'déjà',
                    r'\bDeja\b': 'Déjà',
                    r'\bDEJA\b': 'DÉJÀ',
                    r'\bca\b': 'ça',
                    r'\bCa\b': 'Ça',
                    r'\bCA\b': 'ÇA',
                }
                for pattern, replacement in french_fixes.items():
                    context.output_texts = re.sub(pattern, replacement, context.output_texts)
                
                # Only remove obvious duplicates (same sentence repeated multiple times)
                # Let the LLM work naturally, only clean up obvious errors
                context.output_texts = self._remove_duplicate_endings(context.output_texts)
                
                # French punctuation rules: space before : ; ! ? but not before , .
                context.output_texts = re.sub(r'\s*([,.])\s*', r'\1 ', context.output_texts)  # No space before comma/period
                context.output_texts = re.sub(r'\s*([:;!?])\s*', r' \1 ', context.output_texts)  # Space before : ; ! ?
                context.output_texts = re.sub(r'\s+', ' ', context.output_texts)  # Normalize multiple spaces
                context.output_texts = context.output_texts.strip()
                # Ensure response ends with punctuation if it's a complete sentence
                if context.output_texts and not context.output_texts[-1] in '.!?。！？':
                    context.output_texts += '.'
            
            context.history.add_message(HistoryMessage(role="human", content=chat_text), check_duplicates=False)
            # Check for duplicates before adding avatar response to prevent looping
            context.history.add_message(HistoryMessage(role="avatar", content=context.output_texts), check_duplicates=True)
        except Exception as e:
            logger.error(e)
            if (isinstance(e, APIStatusError)):
                response = e.body
                if isinstance(response, dict) and "message" in response:
                    response = f"{response['message']}"
            output_text = response 
            output = DataBundle(output_definition)
            output.set_main_data(output_text)
            output.add_meta("avatar_text_end", False)
            output.add_meta("speech_id", speech_id)
            yield output
        context.input_texts = ''
        context.output_texts = ''
        logger.info('avatar text end')
        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        end_output.add_meta("avatar_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        yield end_output

    def destroy_context(self, context: HandlerContext):
        pass

