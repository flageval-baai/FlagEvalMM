import requests.models
import json
import requests
import httpx

from typing import Optional, List, Any, Dict
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel
from flagevalmm.prompt.prompt_tools import encode_image

logger = get_logger(__name__)


class HttpClient(BaseApiModel):
    def __init__(
        self,
        model_name: str,
        chat_name: str | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
        url: str | httpx.URL | None = None,
        temperature: float = 0.0,
        stream: bool = False,
        max_image_size: int = 4 * 1024 * 1024,
        min_short_side: int | None = None,
        max_long_side: int | None = None,
        use_cache=False,
    ) -> None:
        super().__init__(
            model_name=model_name,
            chat_name=chat_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            max_image_size=max_image_size,
            min_short_side=min_short_side,
            max_long_side=max_long_side,
            use_cache=use_cache,
        )
        self.chat_args: Dict[str, Any] = {
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if max_tokens is not None:
            self.chat_args["max_tokens"] = max_tokens
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
        }
        if self.url and "azure.com" in self.url.lower():
            self.headers["api-key"] = api_key
        else:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _chat(self, chat_messages: Any, **kwargs):
        data = {"model": f"{self.model_name}", "messages": chat_messages, **kwargs}
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(data), timeout=120
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
            messages[-1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            )
        return messages
