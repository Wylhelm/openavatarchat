

import os
import re
import threading
import time
from typing import Dict, List, Optional, cast
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
from handlers.llm.openai_compatible.chat_history_manager import ChatHistory, HistoryMessage, filter_text, name_dict
from handlers.avatar.emotion.emotion_tag import EmotionTagStreamParser, strip_emotion_tags
from engine_utils.media_utils import ImageUtils


# Removed complex filtering - let the LLM and TTS work naturally


class LLMConfig(HandlerBaseConfigModel, BaseModel):
    model_name: str = Field(default="qwen-plus")
    system_prompt: str = Field(default="请你扮演一个 AI 助手，用简短的对话来回答用户的问题，并在对话内容中加入合适的标点符号，不需要加入标点符号相关的内容")
    api_key: str = Field(default=os.getenv("DASHSCOPE_API_KEY"))
    api_url: str = Field(default=None)
    enable_video_input: bool = Field(default=False)
    video_input_on_demand: bool = Field(default=False)
    voice_elaboration_enabled: bool = Field(default=True)
    history_length: int = Field(default=20)
    temperature: float = Field(default=0.7)  # Lower temperature = more focused, less creative
    max_tokens: int = Field(default=800)  # Allow rich multi-sentence responses when relevant


_VOICE_ELABORATION_INSTRUCTION = (
    "\n\n[CONSIGNE VOCALE PRIORITAIRE — REMPLACE TOUTE LIMITE DE 2-3 PHRASES : "
    "Ton audio est streamé en temps réel dès la première phrase, donc faire une longue réponse "
    "NE ralentit PAS ton délai de réponse initial. "
    "Commence par une première phrase courte et naturelle (5 à 12 mots), puis dès que le sujet s'y prête "
    "(question ouverte, projet, robotique, résumé, explication, conseil, idée, réflexion ou discussion), "
    "DÉVELOPPE spontanément une réponse longue, riche, complète et détaillée de 5 à 8 phrases en style parlé fluide "
    "(sans listes à puces ni markdown). Ne fais une réponse courte (1-2 phrases) QUE pour une simple salutation "
    "ou confirmation rapide.]"
)


_VISUAL_QUERY_PATTERN = re.compile(
    r"\b("
    r"regarde\w*|voi[rst]\w*|vu\b|yeux|oeil|"
    r"cam[eé]ra|webcam|photo|image|[eé]cran|"
    r"montre\w*|tiens|tient|main\w*|doigt\w*|"
    r"porte\w*|habill\w*|v[eê]tement\w*|t-?shirt|chemise|pull|veste|manteau|"
    r"lunette\w*|casquette|chapeau|casque|cheveux|barbe|visage|t[eê]te|souri\w*|grimace|"
    r"air\b|trouves|ressemble\w*|couleur\w*|objet\w*|"
    r"derri[eè]re|devant|autour|fond|pi[eè]ce|chambre|bureau|salon|mur\w*|"
    r"qu['\s]*est[- ]ce que c['\s]*est|qui est[- ]ce|"
    r"look\w*|see\b|seeing|watch\w*|show\w*|holding|wearing|glasses|shirt|behind"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)


def should_include_video_frame(chat_text: str) -> bool:
    if not chat_text:
        return False
    return bool(_VISUAL_QUERY_PATTERN.search(chat_text))


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
        self.shared_states = None
        self.prewarmed_prefix: Optional[List[Dict]] = None
        self.last_prewarm_seq: int = 0
        self.last_conn_warm_ts: float = 0.0
        self.prewarm_thread: Optional[threading.Thread] = None
        self.prewarm_stop = threading.Event()


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
        context.shared_states = session_context.shared_states
        context.client = OpenAI(
            api_key=context.api_key,
            base_url=context.api_url,
            timeout=120.0,
        )
        return context
    
    def start_context(self, session_context, handler_context):
        context = cast(LLMContext, handler_context)
        if context.shared_states is not None and context.prewarm_thread is None:
            context.prewarm_thread = threading.Thread(
                target=self._prewarm_watcher,
                args=(context,),
                daemon=True,
                name=f"llm-prewarm-{context.session_id[:6]}",
            )
            context.prewarm_thread.start()

    @staticmethod
    def _build_prefix_messages(context: LLMContext) -> List[Dict]:
        prefix: List[Dict] = []
        if context.system_prompt and context.system_prompt.get("content"):
            prefix.append(context.system_prompt)
        elif context.system_prompt is not None:
            # Keep deterministic system message structure for KV cache stability
            prefix.append(context.system_prompt)
        if context.history is not None:
            for msg in list(context.history.message_history):
                prefix.append({
                    "role": name_dict[msg.role],
                    "content": filter_text(msg.content),
                })
        return prefix

    def _prewarm_watcher(self, context: LLMContext):
        """Watch for 500ms stable speech signal from VAD to pre-build prompt prefix and warm TCP connection."""
        while not context.prewarm_stop.is_set():
            shared = context.shared_states
            if shared is None or not getattr(shared, "active", True):
                time.sleep(0.05)
                continue
            seq = int(getattr(shared, "prewarm_seq", 0))
            if seq != context.last_prewarm_seq:
                context.last_prewarm_seq = seq
                try:
                    context.prewarmed_prefix = self._build_prefix_messages(context)
                    now = time.monotonic()
                    # Warm HTTP connection pool if idle for > 8s
                    if context.client is not None and (now - context.last_conn_warm_ts) > 8.0:
                        context.last_conn_warm_ts = now
                        try:
                            context.client._client.get("/models", timeout=2.0)
                        except Exception:
                            pass
                    logger.info(
                        f"LLM prewarmed for {getattr(shared, 'prewarm_speech_id', 'turn')} "
                        f"({len(context.prewarmed_prefix)} prefix messages cached)"
                    )
                except Exception as e:
                    logger.debug(f"LLM prewarm error: {e}")
            time.sleep(0.02)

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

    @staticmethod
    def _publish_emotion(context: LLMContext, speech_id: str, emotion: str):
        logger.info(f"Avatar emotion for {speech_id}: {emotion}")
        if context.shared_states is not None:
            context.shared_states.set_avatar_emotion(speech_id, emotion)

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
        video_on_demand = getattr(context.config, 'video_input_on_demand', False)
        include_image = (
            context.enable_video_input
            and context.current_image is not None
            and (not video_on_demand or should_include_video_frame(chat_text))
        )
        if include_image:
            logger.info("Attaching webcam frame to LLM request")

        prefix_messages = (
            context.prewarmed_prefix
            if context.prewarmed_prefix is not None
            else self._build_prefix_messages(context)
        )
        context.prewarmed_prefix = None
        outbound_text = filter_text(chat_text)
        if getattr(context.config, 'voice_elaboration_enabled', True):
            outbound_text = f"{outbound_text}{_VOICE_ELABORATION_INSTRUCTION}"
        if include_image and context.current_image is not None:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": outbound_text},
                    {"type": "image_url", "image_url": {"url": ImageUtils.format_image(context.current_image)}},
                ],
            }
        else:
            user_message = {
                "role": "user",
                "content": outbound_text,
            }
        request_messages = prefix_messages + [user_message]

        tag_parser = EmotionTagStreamParser()
        gen_id = None
        if context.shared_states is not None:
            context.shared_states.active_speech_id = speech_id
            context.shared_states.pipeline_generating = True
            gen_id = getattr(context.shared_states, "turn_generation_id", None)

        aborted = False
        emitted_any_text = False
        try:
            # Get temperature and max_tokens from config if available
            temperature = getattr(context.config, 'temperature', 0.7)
            max_tokens = getattr(context.config, 'max_tokens', 800)
            t_llm_start = time.monotonic()
            context.last_conn_warm_ts = t_llm_start
            completion = context.client.chat.completions.create(
                model=context.model_name,
                messages=request_messages,
                stream=True,
                temperature=temperature,  # Lower temperature = more focused, less creative
                max_tokens=max_tokens,  # Limit response length
                stream_options={"include_usage": True}
            )
            context.current_image = None
            context.input_texts = ''
            context.output_texts = ''
            first_token_marked = False

            for chunk in completion:
                if context.shared_states is not None and hasattr(context.shared_states, "is_turn_aborted"):
                    if context.shared_states.is_turn_aborted(speech_id, gen_id):
                        logger.info(f"LLM stream aborted for {speech_id} due to user barge-in")
                        aborted = True
                        try:
                            completion.close()
                        except Exception:
                            pass
                        break
                if (chunk and chunk.choices and chunk.choices[0] and chunk.choices[0].delta.content):
                    output_text = chunk.choices[0].delta.content
                    if not first_token_marked and context.shared_states is not None:
                        first_token_marked = True
                        if hasattr(context.shared_states, "mark_llm_first_token"):
                            context.shared_states.mark_llm_first_token(speech_id, t_start=t_llm_start)
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

                    output_text, emotion_resolved = tag_parser.feed(output_text)
                    if emotion_resolved:
                        self._publish_emotion(context, speech_id, tag_parser.emotion)
                    if not output_text:
                        continue
                    context.output_texts += output_text
                    # Log if we see "tre" which might indicate "être" was corrupted
                    if 'tre' in output_text.lower() and 'être' not in context.output_texts.lower():
                        logger.warning(f"Possible corruption: 'tre' detected in chunk: '{output_text}'")
                    logger.debug(output_text)
                    output = DataBundle(output_definition)
                    output.set_main_data(output_text)
                    output.add_meta("avatar_text_end", False)
                    output.add_meta("speech_id", speech_id)
                    output.add_meta("avatar_emotion", tag_parser.emotion)
                    emitted_any_text = True
                    yield output

            if not aborted:
                remaining_text, emotion_resolved = tag_parser.flush()
                if emotion_resolved:
                    self._publish_emotion(context, speech_id, tag_parser.emotion)
                if remaining_text:
                    context.output_texts += remaining_text
                    output = DataBundle(output_definition)
                    output.set_main_data(remaining_text)
                    output.add_meta("avatar_text_end", False)
                    output.add_meta("speech_id", speech_id)
                    output.add_meta("avatar_emotion", tag_parser.emotion)
                    emitted_any_text = True
                    yield output

            context.output_texts = strip_emotion_tags(context.output_texts)
            if context.output_texts:
                logger.info(f"LLM response ({speech_id}): {context.output_texts}")

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
            if context.output_texts:
                # Check for duplicates before adding avatar response to prevent looping
                context.history.add_message(HistoryMessage(role="avatar", content=context.output_texts), check_duplicates=True)
        except Exception as e:
            logger.error(e)
            if not aborted:
                response = str(e)
                if (isinstance(e, APIStatusError)):
                    response = e.body
                    if isinstance(response, dict) and "message" in response:
                        response = f"{response['message']}"
                output_text = str(response)
                output = DataBundle(output_definition)
                output.set_main_data(output_text)
                output.add_meta("avatar_text_end", False)
                output.add_meta("speech_id", speech_id)
                emitted_any_text = True
                yield output
        context.input_texts = ''
        context.output_texts = ''
        if aborted:
            return
        if not emitted_any_text and context.shared_states is not None:
            context.shared_states.pipeline_generating = False
            context.shared_states.avatar_speaking = False
            context.shared_states.enable_vad = True
        logger.info('avatar text end')
        end_output = DataBundle(output_definition)
        end_output.set_main_data('')
        end_output.add_meta("avatar_text_end", True)
        end_output.add_meta("speech_id", speech_id)
        end_output.add_meta("avatar_emotion", tag_parser.emotion)
        yield end_output

    def destroy_context(self, context: HandlerContext):
        context = cast(LLMContext, context)
        context.prewarm_stop.set()

