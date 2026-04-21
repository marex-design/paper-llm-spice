from __future__ import annotations

from pathlib import Path
from typing import Dict

from llm.base_client import BaseLLMClient, LLMRequest, LLMResponse


class MockClient(BaseLLMClient):
    def __init__(self, config: Dict) -> None:
        super().__init__(backend_name="mock", model_name="mock-model", config=config)
        self.sample_response_file = Path(config["sample_response_file"])

    def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            text = self.sample_response_file.read_text(encoding="utf-8")
            return LLMResponse(
                success=True,
                text=text,
                backend=self.backend_name,
                model=self.model_name,
                raw_response=text,
                metadata=request.metadata,
            )
        except Exception as exc:
            return LLMResponse(
                success=False,
                text="",
                backend=self.backend_name,
                model=self.model_name,
                raw_response=None,
                error=str(exc),
                metadata=request.metadata,
            )