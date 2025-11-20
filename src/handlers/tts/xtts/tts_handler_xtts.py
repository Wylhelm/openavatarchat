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
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("TTS package not installed. Install with: pip install TTS")


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="tts_models/multilingual/multi-dataset/xtts_v2")
    ref_audio_path: str = Field(default=None)  # Path to reference audio for voice cloning
    language: str = Field(default="fr")  # Language code (fr for French)
    sample_rate: int = Field(default=24000)
    gpu: bool = Field(default=True)  # Use GPU if available
    gpt_cond_len: int = Field(default=3)  # GPT conditioning length (seconds of reference audio to use)


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

        self.model_name = None
        self.ref_audio_path = None
        self.language = None
        self.sample_rate = None
        self.gpu = None
        self.gpt_cond_len = None
        self.tts_model = None

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
        if not TTS_AVAILABLE:
            raise ImportError("TTS package is required. Install with: pip install TTS")
        
        config = cast(TTSConfig, handler_config)
        self.model_name = config.model_name
        self.language = config.language
        self.sample_rate = config.sample_rate
        self.gpu = config.gpu
        self.gpt_cond_len = config.gpt_cond_len
        
        # Handle reference audio path - can be relative to project dir or absolute
        if config.ref_audio_path:
            if os.path.isabs(config.ref_audio_path):
                self.ref_audio_path = config.ref_audio_path
            else:
                self.ref_audio_path = os.path.join(DirectoryInfo.get_project_dir(), config.ref_audio_path)
            
            if not os.path.exists(self.ref_audio_path):
                raise FileNotFoundError(f"Reference audio file not found: {self.ref_audio_path}")
        else:
            raise ValueError("ref_audio_path is required for XTTS voice cloning")
        
        logger.info(f"Loading XTTS model: {self.model_name}")
        logger.info(f"Reference audio: {self.ref_audio_path}")
        logger.info(f"Language: {self.language}")
        
        # Initialize TTS model
        try:
            self.tts_model = TTS(self.model_name, gpu=self.gpu)
            logger.info("XTTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}", exc_info=True)
            raise

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
            if self.tts_model and self.ref_audio_path:
                test_output_path = os.path.join(DirectoryInfo.get_project_dir(), 'temp', f"test_xtts_{context.session_id}.wav")
                os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
                self.tts_model.tts_to_file(
                    text="Test",
                    file_path=test_output_path,
                    speaker_wav=self.ref_audio_path,
                    language=self.language
                )
                if os.path.exists(test_output_path):
                    logger.info("XTTS test successful")
                    os.remove(test_output_path)  # Clean up test file
        except Exception as e:
            logger.warning(f"XTTS test failed: {e}")

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
            logger.info(f"Calling XTTS - Language: {self.language}, Text: '{clean_sentence[:100]}...'")
            
            # Create temporary output file
            temp_output_path = os.path.join(DirectoryInfo.get_project_dir(), 'temp', 
                                           f"xtts_output_{context.session_id}_{int(time.time() * 1000)}.wav")
            os.makedirs(os.path.dirname(temp_output_path), exist_ok=True)
            
            # Generate speech using XTTS
            self.tts_model.tts_to_file(
                text=clean_sentence,
                file_path=temp_output_path,
                speaker_wav=self.ref_audio_path,
                language=self.language
            )
            
            # Load the generated audio
            if not os.path.exists(temp_output_path):
                logger.warning(f"XTTS generated no audio file for sentence: {clean_sentence}")
                return
            
            output_audio, orig_sr = librosa.load(temp_output_path, sr=None)
            
            # Clean up temporary file
            try:
                os.remove(temp_output_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_output_path}: {e}")
            
            # Resample to target sample rate if needed
            if orig_sr != self.sample_rate:
                output_audio = librosa.resample(output_audio, orig_sr=orig_sr, target_sr=self.sample_rate)
            
            # Reshape to match expected format (1, samples)
            output_audio = output_audio[np.newaxis, ...]
            
            output = DataBundle(output_definition)
            output.set_main_data(output_audio)
            output.add_meta("avatar_speech_end", False)
            output.add_meta("speech_id", speech_id)
            logger.info(f"XTTS generated audio: shape={output_audio.shape}, sample_rate={self.sample_rate}")
            context.submit_data(output)
        except Exception as e:
            logger.error(f"Error in XTTS generation for sentence '{clean_sentence}': {e}", exc_info=True)

    def filter_text(self, text):
        if not text:
            return text
        # Preserve ALL French accented characters (é, è, ê, à, ç, etc.) and common punctuation
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
            sentences = re.split(r'(?<=[.!?。！？])\s+', context.input_text)
            if len(sentences) > 1:  # At least one complete sentence
                complete_sentences = sentences[:-1]  # Complete sentences
                context.input_text = sentences[-1]  # Remaining incomplete part

                # Process complete sentences
                for sentence in complete_sentences:
                    if len(sentence.strip()) < 1:
                        continue
                    # Normalize punctuation spacing for French
                    sentence = re.sub(r'\s*([,.])\s*', r'\1 ', sentence)  # No space before comma/period
                    sentence = re.sub(r'\s*([:;!?])\s*', r' \1 ', sentence)  # Space before : ; ! ?
                    sentence = re.sub(r'\s+', ' ', sentence)  # Normalize multiple spaces
                    sentence = sentence.strip()
                    
                    # Remove duplicate words/phrases at the end
                    sentence = self._remove_duplicate_endings(sentence)
                    
                    # Split very long sentences (XTTS works best with reasonable length)
                    max_chunk_length = 4000  # Safe limit
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
                    clean_text = context.input_text.strip()
                    clean_text = re.sub(r'\s*([,.])\s*', r'\1 ', clean_text)  # No space before comma/period
                    clean_text = re.sub(r'\s*([:;!?])\s*', r' \1 ', clean_text)  # Space before : ; ! ?
                    clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize multiple spaces
                    clean_text = clean_text.strip()
                    
                    # Remove duplicate words/phrases at the end
                    clean_text = self._remove_duplicate_endings(clean_text)
                    
                    # Ensure sentence ends with punctuation if it's a complete sentence
                    if clean_text and not clean_text[-1] in '.!?。！？':
                        clean_text += '.'
                    
                    # Split very long sentences
                    max_chunk_length = 4000  # Safe limit
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
                    logger.error(f"Error in XTTS generation for final sentence '{context.input_text}': {e}", exc_info=True)
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

