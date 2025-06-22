from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union, List
import json


@dataclass
class ApiUsage:
    """API usage information containing token counts"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: Optional[Dict[str, Any]] = None
    completion_tokens_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiUsage":
        """Create from dictionary for deserialization"""
        return cls(**data)


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
            "usage": self.usage.to_dict() if self.usage else None,
        }
        return json.dumps(data, ensure_ascii=False)

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
    reason: str = ""
    multiple_raw_answers: Union[List[str], Dict[str, str]] = None
    usage: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Post-initialization processing to ensure multiple_raw_answers has default values"""
        if self.multiple_raw_answers is None:
            if isinstance(self.answer, dict):
                self.multiple_raw_answers = self.answer
            else:
                self.multiple_raw_answers = [self.answer]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility with existing code"""
        return asdict(self)
