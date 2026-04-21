from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from llm.base_client import LLMRequest, LLMResponse
from llm.gemini_client import GeminiClient
from llm.mock_client import MockClient
from llm.openai_client import OpenAIClient
from llm.openai_compatible_client import OpenAICompatibleClient
from llm.deepseek_client import DeepSeekClient  # ← NOUVEAU


class LLMGenerator:
    def __init__(self, root_config: Dict[str, Any]) -> None:
        self.root_config = root_config
        self.llm_config = root_config["llm"]
        self.active_backend = self.llm_config["active_backend"]
        self.defaults = self.llm_config.get("defaults", {})
        self.client = self._build_client()
        
        # Délai entre requêtes pour éviter les rate limits
        self.request_delay = self.defaults.get("request_delay_seconds", 1)

    def _build_client(self):
        if self.active_backend == "gemini":
            return GeminiClient(self.root_config)

        if self.active_backend == "openai":
            return OpenAIClient(self.root_config)

        if self.active_backend == "openai_compat":
            return OpenAICompatibleClient(self.root_config)

        if self.active_backend == "deepseek":  # ← NOUVEAU
            return DeepSeekClient(self.root_config)

        if self.active_backend == "mock":
            return MockClient(self.root_config["mock_backend"])

        raise ValueError(f"Unsupported backend: {self.active_backend}")

    def _build_request(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMRequest:
        return LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature if temperature is not None else self.defaults.get("temperature"),
            top_p=top_p if top_p is not None else self.defaults.get("top_p"),
            top_k=top_k if top_k is not None else self.defaults.get("top_k"),
            max_tokens=max_tokens if max_tokens is not None else self.defaults.get("max_tokens"),
            metadata=metadata or {},
        )

    def generate_one(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        request = self._build_request(
            prompt=prompt,
            system_prompt=system_prompt,
            metadata=metadata,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        return self.client.generate(request)

    def generate_n(
        self,
        prompt: str,
        n: Optional[int] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> List[LLMResponse]:
        total = n if n is not None else self.defaults.get("n_candidates", 1)
        responses: List[LLMResponse] = []

        for idx in range(total):
            item_metadata = dict(metadata or {})
            item_metadata["candidate_index"] = idx + 1
            item_metadata["candidate_id"] = f"cand_{idx + 1:02d}"

            print(f"    [LLM] Request {idx + 1}/{total} (backend: {self.active_backend})...")

            response = self.generate_one(
                prompt=prompt,
                system_prompt=system_prompt,
                metadata=item_metadata,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
            responses.append(response)
            
            if not response.success:
                print(f"    [LLM] Failed: {response.error[:100]}...")
            
            # Délai entre les requêtes pour éviter les rate limits (sauf après la dernière)
            if idx < total - 1:
                time.sleep(self.request_delay)

        return responses

    def get_backend_name(self) -> str:
        return self.client.get_backend_name()

    def get_model_name(self) -> str:
        return self.client.get_model_name()