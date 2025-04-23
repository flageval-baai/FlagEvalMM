import os
from typing import Optional, List, Any, Dict
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel
from flagevalmm.prompt.prompt_tools import encode_image
from flagevalmm.common.video_utils import load_image_or_video
from PIL import Image
import httpx

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
        max_tokens: int = 1024,
        temperature: float = 0.0,
        max_image_size: int = 5 * 1024 * 1024,
        min_short_side: Optional[int] = None,
        max_long_side: int = 8000,
        use_cache: bool = False,
        api_key: Optional[str] = None,
        stream: bool = False,
        use_proxy: bool = False,
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
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if use_proxy:
            proxy_url = os.getenv("PROXY_URL")
            assert proxy_url is not None, "PROXY_URL is not set"
            client = httpx.Client(proxies={"http://": proxy_url, "https://": proxy_url})
        else:
            client = None
        self.client = anthropic.Client(api_key=api_key, http_client=client)

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
                thinking_mode = (
                    "thinking" in chat_args
                    and chat_args["thinking"]["type"] == "enabled"
                )
                begin_thinking = False
                finished_thinking = False
                for event in stream:
                    if event.type == "content_block_start":
                        if not thinking_mode:
                            yield ""
                        elif begin_thinking:
                            yield "\n"
                        else:
                            yield "<think>"
                        begin_thinking = True

                    elif event.type == "content_block_delta":
                        if event.delta.type == "thinking_delta":
                            yield event.delta.thinking
                            # print(f"Thinking: {event.delta.thinking}", end="", flush=True)
                        elif event.delta.type == "text_delta":
                            yield event.delta.text
                            # print(f"Response: {event.delta.text}", end="", flush=True)
                    elif event.type == "content_block_stop":
                        if thinking_mode and begin_thinking and not finished_thinking:
                            yield "</think>"
                            finished_thinking = True
                        else:
                            yield ""
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
        multi_modal_data: Dict[str, Any] = {},
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

        def add_image_to_message(data_path):
            base64_image = encode_image(
                data_path,
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

        for data_type, data_path in multi_modal_data.items():
            if data_type == "image":
                for img_path in data_path:
                    add_image_to_message(img_path)

            elif data_type == "video":
                frames = load_image_or_video(
                    data_path, max_num_frames=self.max_num_frames, return_tensors=False
                )
                for frame in frames:
                    add_image_to_message(Image.fromarray(frame))

        return messages
