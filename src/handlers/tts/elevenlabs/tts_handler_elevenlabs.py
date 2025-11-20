import io
import os
import re
import time
from typing import Dict, Optional, cast
import librosa
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from engine_utils.directory_info import DirectoryInfo

try:
    from elevenlabs import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logger.warning("elevenlabs package not installed. Install with: pip install elevenlabs")


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    api_key: str = Field(default=os.getenv("ELEVENLABS_API_KEY"))
    voice_id: str = Field(default=None)  # ElevenLabs voice ID
    model_id: str = Field(default="eleven_multilingual_v2")  # Model for multilingual support
    sample_rate: int = Field(default=24000)
    stability: float = Field(default=0.5)  # Voice stability (0.0 to 1.0)
    similarity_boost: float = Field(default=0.75)  # Voice similarity boost (0.0 to 1.0)


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.input_text = ''
        self.dump_audio = False
        self.audio_dump_file = None
        self.processed_sentences = set()  # Track processed sentences to avoid duplicates


class HandlerTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

        self.api_key = None
        self.voice_id = None
        self.model_id = None
        self.sample_rate = None
        self.stability = None
        self.similarity_boost = None
        self.client = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=TTSConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, self.sample_rate))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if not ELEVENLABS_AVAILABLE:
            raise ImportError("elevenlabs package is required. Install with: pip install elevenlabs")
        
        config = cast(TTSConfig, handler_config)
        if not config.api_key:
            raise ValueError("ELEVENLABS_API_KEY is required. Set it in config or as environment variable.")
        
        self.api_key = config.api_key
        self.voice_id = config.voice_id
        self.model_id = config.model_id
        self.sample_rate = config.sample_rate
        self.stability = config.stability
        self.similarity_boost = config.similarity_boost
        
        self.client = ElevenLabs(api_key=self.api_key)
        
        if not self.voice_id:
            logger.warning("No voice_id specified. You may need to set one in the config.")
        else:
            logger.info(f"ElevenLabs TTS initialized with voice_id: {self.voice_id}, model: {self.model_id}")

    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, TTSConfig):
            handler_config = TTSConfig()
        context = TTSContext(session_context.session_info.session_id)
        context.input_text = ''
        if context.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), 'temp',
                                            f"dump_avatar_audio_{context.session_id}_{time.localtime().tm_hour}_{time.localtime().tm_min}.pcm")
            context.audio_dump_file = open(dump_file_path, "wb")
        return context
    
    def start_context(self, session_context, context: HandlerContext):
        context = cast(TTSContext, context)
        # Test with a short phrase
        try:
            if self.voice_id:
                test_audio = b''
                for chunk in self.client.text_to_speech.stream(
                    text="Test",
                    voice_id=self.voice_id,
                    model_id=self.model_id,
                    output_format="pcm_24000",
                ):
                    test_audio += chunk
                if len(test_audio) > 0:
                    logger.info("ElevenLabs TTS test successful")
        except Exception as e:
            logger.warning(f"ElevenLabs TTS test failed: {e}")

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
                if len(cleaned_words) >= 6:
                    prev_three = ' '.join(cleaned_words[-6:-3])
                    if last_three == prev_three:
                        cleaned_words = cleaned_words[:-3]
            
            result = ' '.join(cleaned_words)
        
        # Preserve trailing punctuation
        if text.strip() and text.strip()[-1] in '.!?。！？':
            if not result.endswith(text.strip()[-1]):
                result = result.rstrip('.!?。！？') + text.strip()[-1]
        
        return result.strip()

    def _process_sentence_chunk(self, clean_sentence, speech_id, output_definition, context):
        """Process a single sentence chunk for TTS"""
        if not clean_sentence or len(clean_sentence.strip()) < 1:
            return
        
        # Normalize sentence for duplicate detection
        normalized = re.sub(r'\s+', ' ', clean_sentence.lower().strip())
        
        # Check if we've already processed this sentence (or a very similar one)
        if normalized in context.processed_sentences:
            logger.warning(f"Skipping duplicate sentence: '{clean_sentence[:50]}...'")
            return
        
        # Also check for partial matches (if this sentence contains or is contained in a processed one)
        is_duplicate = False
        for processed in context.processed_sentences:
            if len(normalized) > 15 and len(processed) > 15:  # Only check longer sentences
                # Check if one is a substring of the other (with some overlap threshold)
                if normalized in processed or processed in normalized:
                    # Check if they share significant content (at least 80% overlap)
                    words_normalized = set(normalized.split())
                    words_processed = set(processed.split())
                    if len(words_normalized) > 0 and len(words_processed) > 0:
                        overlap = len(words_normalized & words_processed) / max(len(words_normalized), len(words_processed))
                        if overlap > 0.8:  # 80% word overlap
                            is_duplicate = True
                            logger.warning(f"Skipping similar sentence (overlap: {overlap:.2f}): '{clean_sentence[:50]}...'")
                            break
        
        if is_duplicate:
            return
        
        # Add to processed sentences (keep only last 20 to avoid memory issues)
        context.processed_sentences.add(normalized)
        if len(context.processed_sentences) > 20:
            # Remove oldest entries (convert to list, remove first, convert back)
            context.processed_sentences = set(list(context.processed_sentences)[-20:])
        
        logger.info(f'Processing sentence chunk ({len(clean_sentence)} chars): {clean_sentence[:100]}...')
        try:
            logger.info(f"Calling ElevenLabs TTS - Voice: {self.voice_id}, Model: {self.model_id}, Text: '{clean_sentence[:100]}...'")
            
            # Use streaming for better latency
            audio_data = b''
            for chunk in self.client.text_to_speech.stream(
                text=clean_sentence,  # Send text as-is, ElevenLabs will detect language
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format="pcm_24000",
                voice_settings={
                    "stability": self.stability,
                    "similarity_boost": self.similarity_boost,
                }
            ):
                audio_data += chunk
            
            logger.info(f"ElevenLabs TTS generated {len(audio_data)} bytes of audio")
            if len(audio_data) == 0:
                logger.warning(f"ElevenLabs generated no audio data for sentence: {clean_sentence}")
                return
            
            # Convert PCM to numpy array
            # PCM 24000 is 16-bit signed integers
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Reshape to match expected format (1, samples)
            output_audio = audio_array[np.newaxis, ...]
            
            output = DataBundle(output_definition)
            output.set_main_data(output_audio)
            output.add_meta("avatar_speech_end", False)
            output.add_meta("speech_id", speech_id)
            logger.info(f"ElevenLabs TTS generated audio: shape={output_audio.shape}, sample_rate={self.sample_rate}")
            context.submit_data(output)
        except Exception as e:
            logger.error(f"Error in ElevenLabs TTS generation for sentence '{clean_sentence}': {e}", exc_info=True)

    def filter_text(self, text):
        if not text:
            return text
        # Preserve ALL French accented characters (é, è, ê, à, ç, etc.) and common punctuation
        # Explicitly allow French accented characters to ensure they're never removed
        # \w with UNICODE flag should match accented letters, but be explicit
        filtered_text = re.sub(r'[^\w\s,.\-!?;:\'"()àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞŸ，。！？]', '', text, flags=re.UNICODE)
        # Double-check: if "être" becomes "tre", something is wrong
        if 'être' in text and 'être' not in filtered_text:
            logger.warning(f"French character lost in filtering! Original: '{text}', Filtered: '{filtered_text}'")
        return filtered_text

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(TTSContext, context)
        if inputs.type == ChatDataType.AVATAR_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
        speech_id = inputs.data.get_meta("speech_id")
        if (speech_id is None):
            speech_id = context.session_id

        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)
            filtered = self.filter_text(text)
            # Log if French characters might be lost
            if 'ê' in text and 'ê' not in filtered:
                logger.warning(f"Lost 'ê' character! Original: '{text[:50]}', Filtered: '{filtered[:50]}'")
            context.input_text += filtered

        text_end = inputs.data.get_meta("avatar_text_end", False)
        if not text_end:
            # Split on sentence-ending punctuation only (not commas for French)
            # This preserves French sentence structure better
            sentences = re.split(r'(?<=[.!?。！？])\s+', context.input_text)
            if len(sentences) > 1:  # At least one complete sentence
                complete_sentences = sentences[:-1]  # Complete sentences
                context.input_text = sentences[-1]  # Remaining incomplete part

                # Process complete sentences
                for sentence in complete_sentences:
                    if len(sentence.strip()) < 1:
                        continue
                    # Normalize punctuation spacing for French
                    # French rules: space before : ; ! ? but not before , .
                    sentence = re.sub(r'\s*([,.])\s*', r'\1 ', sentence)  # No space before comma/period
                    sentence = re.sub(r'\s*([:;!?])\s*', r' \1 ', sentence)  # Space before : ; ! ?
                    sentence = re.sub(r'\s+', ' ', sentence)  # Normalize multiple spaces
                    sentence = sentence.strip()
                    
                    # Remove duplicate words/phrases at the end (common issue with LLM streaming)
                    sentence = self._remove_duplicate_endings(sentence)
                    
                    # Split very long sentences (ElevenLabs works best with < 5000 chars)
                    max_chunk_length = 4000  # Safe limit for ElevenLabs
                    if len(sentence) > max_chunk_length:
                        # Split on punctuation or commas for very long sentences
                        chunks = re.split(r'([,.;:!?])\s+', sentence)
                        current_chunk = ""
                        for i in range(0, len(chunks), 2):
                            if i < len(chunks):
                                part = chunks[i] + (chunks[i+1] if i+1 < len(chunks) else "")
                                if len(current_chunk) + len(part) > max_chunk_length and current_chunk:
                                    # Process current chunk
                                    self._process_sentence_chunk(current_chunk.strip(), speech_id, output_definition, context)
                                    current_chunk = part
                                else:
                                    current_chunk += part
                        if current_chunk.strip():
                            self._process_sentence_chunk(current_chunk.strip(), speech_id, output_definition, context)
                    else:
                        self._process_sentence_chunk(sentence, speech_id, output_definition, context)
        else:
            logger.info('last sentence: ' + context.input_text)
            if context.input_text is not None and len(context.input_text.strip()) > 0:
                try:
                    # Normalize punctuation spacing for French
                    # French rules: space before : ; ! ? but not before , .
                    clean_text = context.input_text.strip()
                    clean_text = re.sub(r'\s*([,.])\s*', r'\1 ', clean_text)  # No space before comma/period
                    clean_text = re.sub(r'\s*([:;!?])\s*', r' \1 ', clean_text)  # Space before : ; ! ?
                    clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize multiple spaces
                    clean_text = clean_text.strip()
                    
                    # Remove duplicate words/phrases at the end (common issue with LLM streaming)
                    clean_text = self._remove_duplicate_endings(clean_text)
                    
                    # Ensure sentence ends with punctuation if it's a complete sentence
                    if clean_text and not clean_text[-1] in '.!?。！？':
                        clean_text += '.'
                    
                    # Split very long sentences (ElevenLabs works best with < 5000 chars)
                    max_chunk_length = 4000  # Safe limit for ElevenLabs
                    if len(clean_text) > max_chunk_length:
                        # Split on punctuation or commas for very long sentences
                        chunks = re.split(r'([,.;:!?])\s+', clean_text)
                        current_chunk = ""
                        for i in range(0, len(chunks), 2):
                            if i < len(chunks):
                                part = chunks[i] + (chunks[i+1] if i+1 < len(chunks) else "")
                                if len(current_chunk) + len(part) > max_chunk_length and current_chunk:
                                    # Process current chunk
                                    self._process_sentence_chunk(current_chunk.strip(), speech_id, output_definition, context)
                                    current_chunk = part
                                else:
                                    current_chunk += part
                        if current_chunk.strip():
                            self._process_sentence_chunk(current_chunk.strip(), speech_id, output_definition, context)
                    else:
                        self._process_sentence_chunk(clean_text, speech_id, output_definition, context)
                except Exception as e:
                    logger.error(f"Error in ElevenLabs TTS generation for final sentence '{context.input_text}': {e}", exc_info=True)
            context.input_text = ''
            # Clear processed sentences when speech ends to allow new conversation
            context.processed_sentences.clear()
            output = DataBundle(output_definition)
            output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
            output.add_meta("avatar_speech_end", True)
            output.add_meta("speech_id", speech_id)
            context.submit_data(output)
            logger.info(f"speech end")

    def destroy_context(self, context: HandlerContext):
        context = cast(TTSContext, context)
        logger.info('destroy context')

