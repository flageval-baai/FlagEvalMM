import os
import json
import openai
import httpx
from openai import AzureOpenAI, OpenAI

from typing import Optional, List, Any, Union
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel
from flagevalmm.prompt.prompt_tools import encode_image

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
                yield e.message
                return
            else:
                raise e
        if self.stream:
            for chunk in response:
                if len(chunk.choices) and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        else:
            logger.info(f"token num: {response.usage.total_tokens}")
            message = response.choices[0].message
            if hasattr(message, "content"):
                yield message.content
            else:
                yield ""

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
        query, system_prompt=system_prompt, image_paths=["assets/test_1.jpg"]
    )
    answer = model.infer(messages)
