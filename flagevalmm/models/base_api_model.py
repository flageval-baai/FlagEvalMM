import logging
from typing import Optional, Dict, Any, Union, List, Iterator

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)
from flagevalmm.models.model_cache import ModelCache
from flagevalmm.models.api_response import ApiResponse
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
        reasoning: Optional[Dict[str, Any]] = None,
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
        self.chat_args: Dict[str, Any] = {}
        if temperature is not None:
            self.chat_args["temperature"] = temperature
        if reasoning is not None:
            self.chat_args["reasoning"] = reasoning
        if max_tokens is not None:
            self.chat_args["max_tokens"] = max_tokens
        if self.stream:
            self.chat_args["stream"] = True
        self.cache = ModelCache(self.chat_name) if use_cache else None

    def add_to_cache(self, chat_messages, response) -> None:
        """Cache the complete ApiResponse object by serializing it to JSON"""
        if self.cache is None:
            return

        # Serialize ApiResponse to JSON string for database storage
        if isinstance(response, ApiResponse):
            cache_data = response.to_json()
        else:
            # Handle legacy case where response might be a string
            cache_data = str(response)

        self.cache.insert(chat_messages, cache_data)

    def get_from_cache(self, cache_key) -> Optional[ApiResponse]:
        """Get from cache and deserialize back to ApiResponse object"""
        if self.cache is None:
            return None

        result = self.cache.get(cache_key)
        if result is None:
            return None

        # Deserialize JSON string back to ApiResponse object
        try:
            return ApiResponse.from_json(result)
        except Exception as e:
            logger.debug(
                f"Failed to deserialize cached data, treating as legacy string: {e}"
            )
            # Fallback for legacy cached data
            return ApiResponse.from_content(str(result))

    def _chat(self, chat_messages, **kwargs) -> Iterator[Union[ApiResponse, str]]:
        raise NotImplementedError

    @retry(
        wait=wait_random_exponential(multiplier=10, max=100),
        before_sleep=before_sleep_log(logger, logging.INFO),
        stop=stop_after_attempt(5),
    )
    def _single_infer(self, chat_messages, **kwargs) -> ApiResponse:
        final_answer = ""
        final_usage = None
        for res in self._chat(chat_messages, **kwargs):
            if isinstance(res, ApiResponse):
                print(res.content, end="", flush=True)  # noqa T201
                final_answer += res.content
                # Keep the last usage information (most complete)
                if res.usage is not None:
                    final_usage = res.usage
            else:
                # Backward compatibility for non-ApiResponse returns
                print(str(res), end="", flush=True)  # noqa T201
                final_answer += str(res)

        # Return ApiResponse for consistency
        return ApiResponse(content=final_answer, usage=final_usage)

    def infer(self, chat_messages, **kwargs):
        if self.use_cache and self.num_infers == 1:
            cache_key = [chat_messages, kwargs]
            result = self.get_from_cache(cache_key)
            if result is not None:
                logger.info(f"Found in cache\n{result.content}")
                return result

        if self.num_infers == 1:
            final_answer = self._single_infer(chat_messages, **kwargs)
            # Cache the complete ApiResponse object
            if self.use_cache:
                self.add_to_cache([chat_messages, kwargs], final_answer)
            return final_answer
        else:
            logger.info(
                f"Performing {self.num_infers} inferences with temperature {self.temperature}"
            )
            results: List[ApiResponse] = []
            for i in range(self.num_infers):
                logger.info(f"Inference {i+1}/{self.num_infers}")
                if self.use_cache:
                    cache_key = [chat_messages, kwargs, i, self.temperature]
                    result = self.get_from_cache(cache_key)
                    if result is not None:
                        logger.info(f"Found in cache\n{result.content}")
                        results.append(result)
                        continue

                result = self._single_infer(chat_messages, **kwargs)
                results.append(result)
                # Cache the complete ApiResponse object
                if self.use_cache:
                    self.add_to_cache(
                        [chat_messages, kwargs, i, self.temperature], result
                    )

            return results
