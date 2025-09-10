import requests.models
import json
import requests
import httpx
import re
from typing import Optional, List, Any, Union, Dict
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel
from flagevalmm.models.api_response import ApiResponse, ApiUsage
from flagevalmm.prompt.prompt_tools import encode_image
from flagevalmm.common.video_utils import load_image_or_video
from PIL import Image

logger = get_logger(__name__)

IMAGE_REGEX = r"<image \d+>"


class HttpClient(BaseApiModel):
    def __init__(
        self,
        model_name: str,
        chat_name: Optional[str] = None,
        max_tokens: int = 32768,
        temperature: Optional[float] = None,
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        max_num_frames: Optional[int] = 16,
        use_cache: bool = False,
        api_key: Optional[str] = None,
        url: Optional[Union[str, httpx.URL]] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
        retry_time: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            chat_name=chat_name,
            max_tokens=max_tokens,
            temperature=temperature,
            max_image_size=max_image_size,
            min_short_side=min_short_side,
            max_long_side=max_long_side,
            max_num_frames=max_num_frames,
            use_cache=use_cache,
            reasoning=reasoning,
            provider=provider,
            retry_time=retry_time,
            **kwargs,
        )
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        if self.url and "azure.com" in self.url.lower():
            self.headers["api-key"] = api_key
        else:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _chat(self, chat_messages: Any, **kwargs):
        chat_args = self.chat_args.copy()
        chat_args.update(kwargs)
        data = {"model": f"{self.model_name}", "messages": chat_messages, **chat_args}
        if self.stream is False:
            yield from self._non_streaming_chat(data)
        else:
            yield from self._streaming_chat(data)

    def _non_streaming_chat(self, data):
        """Handle non-streaming API requests."""
        if not hasattr(self, "retry_time"):
            retry_time = 300
        else:
            retry_time = self.retry_time

        response = requests.post(
            self.url,
            headers=self.headers,
            data=json.dumps(data),
            timeout=retry_time,
        )
        try:
            response_json = response.json()
        except Exception as e:
            raise Exception(f"Error: {response.text}, {e}")
        if response.status_code != 200:
            if "error" not in response_json:
                yield ApiResponse.from_content(
                    f"Error code: {response_json['message']}"
                )
                return
            err_msg = response_json["error"]
            if "code" in err_msg and "message" in err_msg:
                if (
                    err_msg["code"] == "data_inspection_failed"
                    or err_msg["code"] == "1301"
                    or "no candidates" in err_msg["message"].lower()
                ):
                    yield ApiResponse.from_content(err_msg["message"])
                    return
            raise Exception(
                f"Request failed with status code {response.status_code}: {err_msg}"
            )

        # Parse usage information if available
        usage = None
        if "usage" in response_json:
            usage = ApiUsage.from_dict(response_json["usage"])

        if "choices" in response_json:
            message = response_json["choices"][0]["message"]
            res = ""
            reasoning_content = message.get(
                "reasoning_content", message.get("reasoning", "")
            )
            if reasoning_content:
                res = f"<think>{reasoning_content}</think>\n"
            if "content" in message:
                res += message["content"]
                yield ApiResponse(content=res, usage=usage)
            else:
                yield ApiResponse(content="", usage=usage)
        else:
            yield ApiResponse(
                content=response_json["completions"][0]["text"], usage=usage
            )

    def _streaming_chat(self, data):
        """Handle streaming API requests."""
        think_start = False
        with requests.post(
            self.url,
            headers=self.headers,
            data=json.dumps(data),
            stream=True,
            timeout=300,
        ) as response:
            if response.status_code != 200:
                raise Exception(
                    f"Stream request failed with status code {response.status_code}: {response.text}"
                )

            for line in response.iter_lines():
                if line:
                    # Remove "data: " prefix if it exists (common in SSE)
                    line_text = line.decode("utf-8")
                    if '"usage":null' not in line_text:
                        print(f"line_text: {line_text}")
                    if line_text.startswith("data: "):
                        line_text = line_text[6:]

                    # Skip heartbeat or empty messages
                    if line_text.strip() == "" or line_text == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(line_text)
                        # Extract content from the chunk based on API response format
                        if "choices" in chunk:
                            delta = chunk["choices"][0].get("delta", {})
                            if (
                                "reasoning_content" in delta
                                and delta["reasoning_content"]
                            ):
                                content = delta["reasoning_content"]
                                if think_start is False:
                                    think_start = True
                                    content = f"<think>{content}"
                                yield ApiResponse.from_content(content)
                            if "content" in delta and delta["content"]:
                                content = delta["content"]
                                if think_start:
                                    content = f"</think>\n{content}"
                                    think_start = False
                                if chunk.get("usage") is not None:
                                    usage = ApiUsage.from_dict(chunk["usage"])
                                    yield ApiResponse(content=content, usage=usage)
                                else:
                                    yield ApiResponse.from_content(content)
                    except json.JSONDecodeError as e:
                        raise Exception(
                            f"Failed to parse chunk: {line_text}, error: {e}"
                        )

    def add_image_to_message(self, image_data, messages):
        base64_image = encode_image(
            image_data,
            max_size=self.max_image_size,
            min_short_side=self.min_short_side,
            max_long_side=self.max_long_side,
        )
        messages[-1]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )

    def build_message(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        multi_modal_data: Dict[str, Any] = {},
        past_messages: Optional[List] = None,
    ) -> List:
        messages = past_messages if past_messages else []
        system_prompt = system_prompt if system_prompt else self.system_prompt

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if not multi_modal_data:
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
        for data_type, data_list in multi_modal_data.items():
            # TODO: merge video and image
            if data_type == "image":
                self.build_interleaved_message(query, messages, data_list)
            elif data_type == "video":
                self.build_video_message(query, messages, data_list)
        return messages

    def build_video_message(self, query: str, messages: List, video_data: str):
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
        frames = load_image_or_video(
            video_data, max_num_frames=self.max_num_frames, return_tensors=False
        )
        for frame in frames:
            self.add_image_to_message(Image.fromarray(frame), messages)

    def build_interleaved_message(
        self, query: str, messages: List, image_data: List[Union[str, Image.Image]]
    ):
        referenced_numbers = [
            int(re.search(r"\d+", ref).group())
            for ref in re.findall(IMAGE_REGEX, query)
        ]
        content = []
        # Check if all referenced numbers are valid
        if referenced_numbers:
            max_ref = max(referenced_numbers)
            min_ref = min(referenced_numbers)
            if max_ref > len(image_data) or min_ref < 1:
                raise ValueError("Invalid image reference in question.")

        base64_images = [
            encode_image(
                data,
                max_size=self.max_image_size,
                min_short_side=self.min_short_side,
                max_long_side=self.max_long_side,
            )
            for data in image_data
        ]

        parts = re.split(r"(<image \d+>)", query)
        for part in parts:
            if len(part.strip()) == 0:
                continue
            if re.match(IMAGE_REGEX, part):
                # It's an image reference
                num = int(re.search(r"\d+", part).group())
                base64_image = base64_images[num - 1]
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )
            else:
                assert len(part.strip()) > 0, f"part: {part}"
                # It's a text part
                content.append({"type": "text", "text": part})
        # If there are no referenced images, add all images to the message, not interleaved
        if not referenced_numbers:
            for base64_image in base64_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )
        messages.append({"role": "user", "content": content})
