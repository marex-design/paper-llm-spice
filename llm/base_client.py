from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMRequest:
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    success: bool
    text: str
    backend: str
    model: str
    raw_response: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLMClient(ABC):
    def __init__(self, backend_name: str, model_name: str, config: Dict[str, Any]) -> None:
        self.backend_name = backend_name
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def get_backend_name(self) -> str:
        return self.backend_name

    def get_model_name(self) -> str:
        return self.model_name