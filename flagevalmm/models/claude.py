import os
from typing import Optional, List, Any, Dict
from flagevalmm.common.logger import get_logger
from flagevalmm.models.api_response import ApiResponse, ApiUsage
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
        max_tokens: int = 64000,
        temperature: float = 0.0,
        max_image_size: int = 5 * 1024 * 1024,
        min_short_side: Optional[int] = None,
        max_long_side: int = 8000,
        use_cache: bool = False,
        api_key: Optional[str] = None,
        stream: bool = False,
        use_proxy: bool = False,
        thinking: Optional[Dict[str, Any]] = None,
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
        if thinking is not None:
            self.chat_args["thinking"] = thinking
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
                thinking_started = False
                thinking_text = ""
                answer_text = ""
                input_tokens = output_tokens = 0

                for event in stream:
                    if event.type == "content_block_start":
                        if thinking_mode and not thinking_started:
                            thinking_text += "<think>"
                        thinking_started = True

                    elif event.type == "content_block_delta":
                        if event.delta.type == "thinking_delta":
                            thinking_text += event.delta.thinking
                        elif event.delta.type == "text_delta":
                            answer_text += event.delta.text

                    elif event.type == "content_block_stop":
                        if (
                            thinking_mode
                            and thinking_started
                            and not thinking_text.endswith("</think>")
                        ):
                            thinking_text += "</think>"

                    elif event.type == "message_delta":
                        output_tokens = event.usage.output_tokens
                    elif event.type == "message_start":
                        input_tokens = event.message.usage.input_tokens

                yield ApiResponse(
                    content=thinking_text + answer_text,
                    usage=ApiUsage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                        prompt_tokens_details=None,
                        completion_tokens_details=None,
                    ),
                )
        else:
            response = self.client.messages.create(
                system=system_prompt,
                messages=chat_messages,
                model=self.model_name,
                **chat_args,
            )
            api_response = ApiResponse(
                content=response.content[0].text,
                usage=ApiUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.output_tokens
                    + response.usage.input_tokens,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                ),
            )
            yield api_response

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
