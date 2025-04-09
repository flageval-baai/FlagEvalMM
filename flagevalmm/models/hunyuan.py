import os
import json
import types
import httpx
from typing import Optional, List, Any, Dict
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel
from flagevalmm.prompt.prompt_tools import encode_image
from flagevalmm.common.video_utils import load_image_or_video
from PIL import Image

logger = get_logger(__name__)

credential: Any = None
ClientProfile: Any = None
HttpProfile: Any = None
hunyuan_client: Any = None
models: Any = None


def _load_tencent_packages():
    global credential, ClientProfile, HttpProfile, hunyuan_client, models
    try:
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
    except Exception:
        logger.error(
            "Tencent Cloud SDK for Python is not installed, run `pip install tencentcloud-sdk-python`"
        )
        raise


class Hunyuan(BaseApiModel):
    def __init__(
        self,
        model_name: str,
        url: str | httpx.URL | None = None,
        chat_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        stream: bool = False,
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        max_num_frames: Optional[int] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        _load_tencent_packages()
        super().__init__(
            model_name=model_name,
            chat_name=chat_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            use_cache=use_cache,
            max_image_size=max_image_size,
            min_short_side=min_short_side,
            max_long_side=max_long_side,
            max_num_frames=max_num_frames,
        )
        self.model_type = "hunyuan"
        assert os.getenv("HUNYUAN_AK") and os.getenv(
            "HUNYUAN_SK"
        ), "HUNYUAN_AK and HUNYUAN_SK must be set"
        cred = credential.Credential(os.getenv("HUNYUAN_AK"), os.getenv("HUNYUAN_SK"))
        httpProfile = HttpProfile()
        httpProfile.endpoint = url
        httpProfile.reqTimeout = 300

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.client = hunyuan_client.HunyuanClient(cred, "", clientProfile)

    def _chat(self, chat_messages: Any, **kwargs):
        req = models.ChatCompletionsRequest()
        params = {"Model": self.model_name, "Messages": chat_messages}

        req.from_json_string(json.dumps(params))
        try:
            resp = self.client.ChatCompletions(req)
        except Exception as e:
            raise Exception(f"Error in Hunyuan API: {e}")
        if isinstance(resp, types.GeneratorType):  # stream response
            for event in resp:
                yield event
        else:  # non-stream response
            yield resp.Choices[0].Message.Content

    def build_message(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        multi_modal_data: Dict[str, Any] = {},
        past_messages: Optional[List] = None,
    ) -> List:
        messages = past_messages if past_messages else []
        if system_prompt:
            messages.append(
                {
                    "Role": "system",
                    "Contents": [{"Type": "text", "Text": system_prompt}],
                }
            )
        messages.append({"Role": "user", "Contents": [{"Type": "text", "Text": query}]})

        def add_image_to_message(data_path):
            base64_image = encode_image(
                data_path,
                max_size=self.max_image_size,
                min_short_side=self.min_short_side,
                max_long_side=self.max_long_side,
            )
            messages[-1]["Contents"].append(
                {
                    "Type": "image_url",
                    "ImageUrl": {"Url": f"data:image/jpeg;base64,{base64_image}"},
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
