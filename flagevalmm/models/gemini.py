import os
from PIL import Image
from typing import Optional, List, Any
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel

logger = get_logger(__name__)

# Lazy loading of Google packages
genai: Any = None
glm: Any = None
HarmBlockThreshold: Any = None
HarmCategory: Any = None


def _load_google_packages():
    global genai, glm, HarmBlockThreshold, HarmCategory
    try:
        import google.generativeai as genai
        from google.ai import generativelanguage as glm
        from google.generativeai.types import HarmBlockThreshold, HarmCategory
    except Exception:
        logger.error(
            "google-generativeai is not installed, please install it by `pip install google-generativeai`"
        )
        raise


class Gemini(BaseApiModel):
    def __init__(
        self,
        model_name: str,
        chat_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        use_cache: bool = False,
        api_key: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> None:
        _load_google_packages()
        super().__init__(
            model_name=model_name,
            chat_name=chat_name,
            max_tokens=max_tokens,
            temperature=temperature,
            max_image_size=max_image_size,
            min_short_side=min_short_side,
            max_long_side=max_long_side,
            use_cache=use_cache,
        )
        self.model_type = "gemini"
        api_key = api_key if api_key else os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model_name)

    def _chat(self, chat_messages: Any, **kwargs):
        config = genai.GenerationConfig(
            max_output_tokens=self.max_tokens, temperature=self.temperature
        )
        response = self.client.generate_content(
            chat_messages,
            generation_config=config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        if (
            response.prompt_feedback.block_reason
            == glm.GenerateContentResponse.PromptFeedback.BlockReason.OTHER
        ):
            yield "Can not answer because of blocked."
            return

        finish_reason = response.candidates[0].finish_reason
        if (
            finish_reason == glm.Candidate.FinishReason.SAFETY
            or finish_reason == glm.Candidate.FinishReason.OTHER
        ):
            yield "Can not answer because of safety reasons."
            return
        yield response.text

    def build_message(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        image_paths: List[str] = [],
        past_messages: Optional[List] = None,
    ) -> List:
        messages = past_messages if past_messages else []
        if system_prompt:
            messages.append(system_prompt)
        messages.append(query)
        for img_path in image_paths:
            im = Image.open(img_path).convert("RGB")
            messages.append(im)
        return messages
