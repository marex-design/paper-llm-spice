from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from google import genai
from google.genai import types

from llm.base_client import BaseLLMClient, LLMRequest, LLMResponse


class GeminiClient(BaseLLMClient):
    def __init__(self, root_config: Dict[str, Any]) -> None:
        llm_cfg = root_config["llm"]
        backend_cfg = root_config["backends"]["gemini"]

        model_name = backend_cfg["model"]["name"]
        super().__init__(backend_name="gemini", model_name=model_name, config=backend_cfg)

        self.root_config = root_config
        self.llm_cfg = llm_cfg
        self.backend_cfg = backend_cfg
        self.defaults = llm_cfg.get("defaults", {})
        self.request_cfg = backend_cfg.get("request", {})
        self.response_cfg = llm_cfg.get("response_handling", {})
        self.logging_cfg = root_config.get("logging", {})
        self.prompting_cfg = llm_cfg.get("prompting", {})

        api_key_env_var = backend_cfg["authentication"]["api_key_env_var"]
        api_key = os.getenv(api_key_env_var)

        if not api_key:
            raise ValueError(f"Missing API key in environment variable: {api_key_env_var}")

        self.client = genai.Client(api_key=api_key)

        self.max_retries = self._get_cfg_value("max_retries", default=2)
        self.retry_delay_seconds = self._get_cfg_value("retry_delay_seconds", default=2)
        self.timeout_seconds = self._get_cfg_value("timeout_seconds", default=60)

    def generate(self, request: LLMRequest) -> LLMResponse:
        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 2):
            try:
                prompt = self._build_prompt(
                    system_prompt=request.system_prompt,
                    user_prompt=request.prompt,
                )

                generation_config = self._build_generation_config(request)

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generation_config,
                )

                text = self._extract_text(response)

                return LLMResponse(
                    success=True,
                    text=text,
                    backend=self.backend_name,
                    model=self.model_name,
                    raw_response=response,
                    error=None,
                    metadata={
                        **request.metadata,
                        "attempt": attempt,
                        "timeout_seconds": self.timeout_seconds,
                        "request_config": self._export_effective_request_config(request),
                    },
                )

            except Exception as exc:
                last_error = str(exc)

                if attempt <= self.max_retries:
                    time.sleep(self.retry_delay_seconds)
                else:
                    break

        return LLMResponse(
            success=False,
            text="",
            backend=self.backend_name,
            model=self.model_name,
            raw_response=None,
            error=last_error or "Unknown Gemini generation error",
            metadata={
                **request.metadata,
                "attempts": self.max_retries + 1,
                "timeout_seconds": self.timeout_seconds,
                "request_config": self._export_effective_request_config(request),
            },
        )

    def _get_cfg_value(self, key: str, default: Any = None) -> Any:
        if key in self.request_cfg:
            return self.request_cfg[key]
        if key in self.defaults:
            return self.defaults[key]
        return default

    def _build_prompt(self, system_prompt: Optional[str], user_prompt: str) -> str:
        include_system_prompt = self.prompting_cfg.get("include_system_prompt", True)

        if include_system_prompt and system_prompt and system_prompt.strip():
            return f"{system_prompt.strip()}\n\n{user_prompt.strip()}"

        return user_prompt.strip()

    def _build_generation_config(self, request: LLMRequest) -> types.GenerateContentConfig:
        temperature = request.temperature if request.temperature is not None else self._get_cfg_value("temperature", 0.2)
        top_p = request.top_p if request.top_p is not None else self._get_cfg_value("top_p", 0.8)
        top_k = request.top_k if request.top_k is not None else self._get_cfg_value("top_k", 20)
        max_tokens = request.max_tokens if request.max_tokens is not None else self._get_cfg_value("max_output_tokens", self._get_cfg_value("max_tokens", 4096))

        return types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
        )

    def _export_effective_request_config(self, request: LLMRequest) -> Dict[str, Any]:
        return {
            "temperature": request.temperature if request.temperature is not None else self._get_cfg_value("temperature", 0.2),
            "top_p": request.top_p if request.top_p is not None else self._get_cfg_value("top_p", 0.8),
            "top_k": request.top_k if request.top_k is not None else self._get_cfg_value("top_k", 20),
            "max_output_tokens": request.max_tokens if request.max_tokens is not None else self._get_cfg_value("max_output_tokens", self._get_cfg_value("max_tokens", 4096)),
            "n_candidates_per_prompt": self._get_cfg_value("n_candidates_per_prompt", 1),
            "timeout_seconds": self.timeout_seconds,
        }

    def _extract_text(self, response: Any) -> str:
        if hasattr(response, "text") and response.text:
            return response.text

        candidates = getattr(response, "candidates", None)
        if candidates:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                if not content:
                    continue

                parts = getattr(content, "parts", None)
                if not parts:
                    continue

                collected = []
                for part in parts:
                    text = getattr(part, "text", None)
                    if text:
                        collected.append(text)

                if collected:
                    return "\n".join(collected).strip()

        raise ValueError("Unable to extract text from Gemini response")