import os
import json
import openai
import httpx
from openai import AzureOpenAI, OpenAI
from PIL import Image

from typing import Optional, List, Any, Union, Dict
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel
from flagevalmm.models.api_response import ApiResponse, ApiUsage
from flagevalmm.prompt.prompt_tools import encode_image_old
from flagevalmm.common.video_utils import load_image_or_video

logger = get_logger(__name__)


class GPT(BaseApiModel):
    def __init__(
        self,
        model_name: str,
        chat_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        max_num_frames: Optional[int] = None,
        use_cache: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[Union[str, httpx.URL]] = None,
        stream: bool = False,
        use_azure_api: bool = False,
        json_mode: bool = False,
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
        self.model_type = "gpt"
        if json_mode:
            self.chat_args["response_format"] = {"type": "json_object"}

        if use_azure_api:
            if api_key is None:
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2023-12-01-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        else:
            if api_key is None:
                api_key = os.getenv("BAAI_OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _chat(self, chat_messages: Any, **kwargs):
        try:
            chat_args = self.chat_args.copy()
            chat_args.update(kwargs)
            response = self.client.chat.completions.create(
                model=self.model_name, messages=chat_messages, **chat_args
            )
        except openai.APIStatusError as e:
            if e.status_code == openai.BadRequestError or e.status_code == 451:
                yield ApiResponse.from_content(e.message)
                return
            else:
                raise e
        if self.stream:
            for chunk in response:
                if len(chunk.choices) and chunk.choices[0].delta.content is not None:
                    yield ApiResponse.from_content(chunk.choices[0].delta.content)
        else:
            logger.info(f"token num: {response.usage.total_tokens}")

            # Parse usage information
            usage = None
            if response.usage:
                usage = ApiUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    # OpenAI response may not have these detailed fields
                    prompt_tokens_details=getattr(
                        response.usage, "prompt_tokens_details", None
                    ),
                    completion_tokens_details=getattr(
                        response.usage, "completion_tokens_details", None
                    ),
                )

            message = response.choices[0].message
            if hasattr(message, "content"):
                yield ApiResponse(content=message.content, usage=usage)
            else:
                yield ApiResponse(content="", usage=usage)

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

        content = [{"type": "text", "text": query}]

        def add_image_to_message(data_path):
            base64_image = encode_image_old(
                data_path,
                max_size=self.max_image_size,
                min_short_side=self.min_short_side,
                max_long_side=self.max_long_side,
            )
            content.append(
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

        messages.append({"role": "user", "content": content})
        return messages

    def get_embedding(self, text: str):
        if self.cache is not None:
            response = self.cache.get(text)
            if response:
                return json.loads(response)
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text],
        )
        self.add_to_cache(text, str(response.data[0].embedding))  # type: ignore
        return response.data[0].embedding  # type: ignore


if __name__ == "__main__":
    model = GPT(
        model_name="gpt-4o-mini",
        temperature=0.5,
        use_cache=False,
        stream=True,
    )
    query = "给我说一个关于火星的笑话，要非常好笑"
    system_prompt = "Your task is to generate a joke that is very funny."
    messages = model.build_message(query, system_prompt=system_prompt)
    answer = model.infer(messages)

    query = "根据这张图片的内容，写一个笑话"
    messages = model.build_message(
        query,
        system_prompt=system_prompt,
        multi_modal_data={"image": ["assets/test_1.jpg"]},
    )
    answer = model.infer(messages)
