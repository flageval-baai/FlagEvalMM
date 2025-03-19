import requests.models
import json
import requests
import httpx

from typing import Optional, List, Any, Union, Dict
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel
from flagevalmm.prompt.prompt_tools import encode_image
from flagevalmm.common.video_utils import load_image_or_video
from PIL import Image

logger = get_logger(__name__)


class HttpClient(BaseApiModel):
    def __init__(
        self,
        model_name: str,
        chat_name: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        max_num_frames: Optional[int] = 16,
        use_cache: bool = False,
        api_key: Optional[str] = None,
        url: Optional[Union[str, httpx.URL]] = None,
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
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(data), timeout=300
        )
        response_json = response.json()
        if response.status_code != 200:
            if "error" not in response_json:
                yield f"Error code: {response_json['message']}"
                return
            err_msg = response_json["error"]
            if "code" in err_msg and (
                err_msg["code"] == "data_inspection_failed" or err_msg["code"] == "1301"
            ):
                yield err_msg["message"]
                return
            raise Exception(
                f"Request failed with status code {response.status_code}: {err_msg}"
            )
        if "choices" in response_json:
            message = response_json["choices"][0]["message"]
            if "content" in message:
                yield message["content"]
            else:
                yield ""
        else:
            yield response_json["completions"][0]["text"]

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

        def add_image_to_message(image_data):
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
