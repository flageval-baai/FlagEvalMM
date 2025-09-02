from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union, List
import json


@dataclass
class PromptTokensDetails:
    """
    Prompt token details
    """
    text_tokens: Optional[int] = 0
    audio_tokens: Optional[int] = 0
    cached_tokens: Optional[int] = 0
    image_tokens: Optional[int] = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTokensDetails":
        """Create from dictionary for deserialization"""
        allowed_keys = {"text_tokens", "audio_tokens", "cached_tokens", "image_tokens"}
        filtered = {k: data[k] for k in allowed_keys if k in data}
        return cls(**filtered)


@dataclass
class CompletionTokensDetails:
    """
    Completion token details
    """

    accepted_prediction_tokens: int = 0
    audio_tokens: Optional[int] = 0
    text_tokens: Optional[int] = 0
    reasoning_tokens: int = 0
    rejected_prediction_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompletionTokensDetails":
        """Create from dictionary for deserialization"""
        allowed_keys = {
            "accepted_prediction_tokens",
            "audio_tokens",
            "text_tokens",
            "reasoning_tokens",
            "rejected_prediction_tokens",
        }
        filtered = {k: data[k] for k in allowed_keys if k in data}
        return cls(**filtered)


# Custom JSON encoder for dataclasses
class DataclassJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for dataclasses"""

    def default(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


@dataclass
class ApiUsage:
    """API usage information containing token counts"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: Union[PromptTokensDetails, Dict[str, Any]] = None
    completion_tokens_details: Union[CompletionTokensDetails, Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Ensure nested dataclasses are converted to dictionaries
        if isinstance(self.prompt_tokens_details, PromptTokensDetails):
            result["prompt_tokens_details"] = self.prompt_tokens_details.to_dict()
        if isinstance(self.completion_tokens_details, CompletionTokensDetails):
            result["completion_tokens_details"] = (
                self.completion_tokens_details.to_dict()
            )
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiUsage":
        """Create from dictionary for deserialization"""
        # Only keep allowed keys to avoid initialization errors
        allowed_keys = {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
        }

        filtered: Dict[str, Any] = {k: data[k] for k in allowed_keys if k in data}

        # Handle nested dataclass objects safely
        prompt_details = data.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            filtered["prompt_tokens_details"] = PromptTokensDetails.from_dict(prompt_details)
        elif isinstance(prompt_details, PromptTokensDetails):
            filtered["prompt_tokens_details"] = prompt_details

        completion_details = data.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            filtered["completion_tokens_details"] = CompletionTokensDetails.from_dict(completion_details)
        elif isinstance(completion_details, CompletionTokensDetails):
            filtered["completion_tokens_details"] = completion_details

        return cls(**filtered)


@dataclass
class ApiResponse:
    """API response containing content and usage information"""

    content: str
    usage: Optional[ApiUsage] = None

    @classmethod
    def from_content(cls, content: str) -> "ApiResponse":
        """Create ApiResponse with only content (for backward compatibility)"""
        return cls(content=content, usage=None)

    def to_json(self) -> str:
        """Serialize to JSON string for caching"""
        data = {
            "content": self.content,
            "usage": self.usage,  # No need to convert explicitly, the encoder will handle it
        }
        return json.dumps(data, ensure_ascii=False, cls=DataclassJSONEncoder)

    @classmethod
    def from_json(cls, json_str: str) -> "ApiResponse":
        """Deserialize from JSON string for caching"""
        try:
            data = json.loads(json_str)
            usage = None
            if data.get("usage"):
                usage = ApiUsage.from_dict(data["usage"])
            return cls(content=data["content"], usage=usage)
        except (json.JSONDecodeError, KeyError, TypeError):
            # If JSON parsing fails, treat as legacy string content
            return cls.from_content(json_str)


@dataclass
class ProcessResult:
    """
    Data class representing the result of processing a single item
    """

    question_id: str
    question: str
    answer: Union[
        str, Dict[str, str]
    ]  # Can be string or multiple inference results dictionary
    reason: Union[str, List[str]] = (
        ""  # Can be string or list of strings for multiple inferences
    )
    usage: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = (
        None  # Can be single usage or list of usages
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility with existing code"""
        return asdict(self)
