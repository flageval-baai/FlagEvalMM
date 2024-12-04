import logging
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)
from flagevalmm.models.model_cache import ModelCache
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)


class BaseApiModel:
    def __init__(
        self,
        model_name: str,
        chat_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        stream: bool = False,
        max_image_size: int = 4 * 1024 * 1024,
        min_short_side: int = 28,
        max_long_side: int = 1500,
        use_cache=False,
    ) -> None:
        self.model_name = model_name
        self.chat_name = chat_name if chat_name else model_name
        self.use_cache = use_cache
        self.model_type = "base"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stream = stream
        self.max_image_size = max_image_size
        self.min_short_side = min_short_side
        self.max_long_side = max_long_side
        self.cache = ModelCache(self.chat_name) if use_cache else None

    def add_to_cache(self, chat_messages, response) -> None:
        if self.cache is None:
            return
        self.cache.insert(chat_messages, response)

    def _chat(self, chat_messages, **kwargs):
        raise NotImplementedError

    @retry(
        wait=wait_random_exponential(multiplier=10, max=100),
        before_sleep=before_sleep_log(logger, logging.INFO),
        stop=stop_after_attempt(5),
    )
    def infer(self, chat_messages, **kwargs):
        if self.use_cache:
            result = self.cache.get([chat_messages, kwargs])
            if result:
                logger.info(f"Found in cache\n{result}")
                return result

        final_answer = ""
        for res in self._chat(chat_messages, **kwargs):
            print(res, end="", flush=True)  # noqa T201
            final_answer += res
        print()  # noqa T201
        self.add_to_cache([chat_messages, kwargs], final_answer)
        return final_answer
