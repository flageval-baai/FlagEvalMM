from typing import Optional, List, Any
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel
from flagevalmm.prompt.prompt_tools import encode_image

logger = get_logger(__name__)

# Lazy loading of Anthropic packages
anthropic: Any = None


def _load_anthropic_packages():
    global anthropic
    try:
        import anthropic
    except Exception:
        logger.error(
            "Anthropic SDK for Python is not installed, run `pip install anthropic`"
        )
        raise


class Claude(BaseApiModel):
    def __init__(
        self,
        model_name: str,
        chat_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        max_image_size: int = 5 * 1024 * 1024,
        min_short_side: Optional[int] = None,
        max_long_side: int = 8000,
        use_cache: bool = False,
        api_key: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> None:
        _load_anthropic_packages()
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
        self.model_type = "claude"
        self.stream = stream
        self.client = anthropic.Anthropic(api_key=api_key)

    def _chat(self, chat_messages: Any, **kwargs):
        system_prompt = (
            chat_messages.pop(0)["content"]
            if chat_messages[0]["role"] == "system"
            else anthropic._types.NOT_GIVEN
        )
        chat_args = self.chat_args.copy()
        chat_args.update(kwargs)
        if self.stream:
            with self.client.messages.stream(
                system=system_prompt,
                messages=chat_messages,
                model=self.model_name,
                **chat_args,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        else:
            response = self.client.messages.create(
                system=system_prompt,
                messages=chat_messages,
                model=self.model_name,
                **chat_args,
            )
            yield response.content[0].text

    def build_message(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        image_paths: List[str] = [],
        past_messages: Optional[List] = None,
    ) -> List:
        messages = past_messages if past_messages else []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    },
                ],
            },
        )
        for img_path in image_paths:
            base64_image = encode_image(
                img_path,
                max_size=self.max_image_size,
                min_short_side=self.min_short_side,
                max_long_side=self.max_long_side,
            )

            media_type = "image/jpeg"
            messages[-1]["content"].append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image,
                    },
                },
            )
        return messages
