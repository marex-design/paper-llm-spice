from __future__ import annotations

from typing import Dict

from llm.base_client import BaseLLMClient, LLMRequest, LLMResponse


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: Dict) -> None:
        model_name = config["model"]["name"]
        super().__init__(backend_name="openai", model_name=model_name, config=config)

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            success=False,
            text="",
            backend=self.backend_name,
            model=self.model_name,
            raw_response=None,
            error="OpenAI backend not implemented yet",
            metadata=request.metadata,
        )