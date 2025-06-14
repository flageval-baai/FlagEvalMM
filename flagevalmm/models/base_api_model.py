import logging
from typing import Optional, Dict, Any

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
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        max_num_frames: Optional[int] = 16,
        use_cache: bool = False,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        num_infers: int = 1,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.chat_name = chat_name if chat_name else model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_image_size = max_image_size
        self.min_short_side = min_short_side
        self.max_long_side = max_long_side
        self.max_num_frames = max_num_frames
        self.use_cache = use_cache
        self.model_type = "base"
        self.stream = stream
        self.system_prompt = system_prompt
        self.num_infers = num_infers
        if num_infers > 1:
            if temperature == 0:
                logger.warning("set temperature to 1")
                temperature = 1
        self.chat_args: Dict[str, Any] = {
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            self.chat_args["max_tokens"] = max_tokens
        if self.stream:
            self.chat_args["stream"] = True
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
    def _single_infer(self, chat_messages, **kwargs):
        final_answer = ""
        for res in self._chat(chat_messages, **kwargs):
            print(res, end="", flush=True)  # noqa T201
            final_answer += res
        return final_answer

    def infer(self, chat_messages, **kwargs):
        if self.use_cache and self.num_infers == 1:
            result = self.cache.get([chat_messages, kwargs])
            if result:
                logger.info(f"Found in cache\n{result}")
                return result

        if self.num_infers == 1:
            final_answer = self._single_infer(chat_messages, **kwargs)
            self.add_to_cache([chat_messages, kwargs], final_answer)
            return final_answer
        else:
            logger.info(
                f"Performing {self.num_infers} inferences with temperature {self.temperature}"
            )
            results = []
            for i in range(self.num_infers):
                logger.info(f"Inference {i+1}/{self.num_infers}")
                if self.use_cache:
                    result = self.cache.get(
                        [chat_messages, kwargs, i, self.temperature]
                    )
                    if result:
                        logger.info(f"Found in cache\n{result}")
                        results.append(result)
                        continue
                result = self._single_infer(chat_messages, **kwargs)
                results.append(result)
                self.add_to_cache([chat_messages, kwargs, i, self.temperature], result)

            return results
