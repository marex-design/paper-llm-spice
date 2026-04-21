from __future__ import annotations

import os
from typing import Any, Dict, List

from openai import OpenAI

from llm.base_client import BaseLLMClient, LLMRequest, LLMResponse


class DeepSeekClient(BaseLLMClient):
    """Client pour l'API DeepSeek (utilise le SDK OpenAI)."""

    def __init__(self, root_config: Dict[str, Any]) -> None:
        self.root_config = root_config
        llm_cfg = root_config["llm"]
        backend_name = llm_cfg["active_backend"]
        backend_cfg = root_config["backends"][backend_name]
        model_name = backend_cfg["model"]["name"]

        super().__init__(backend_name, model_name, backend_cfg)

        auth_cfg = backend_cfg.get("authentication", {})
        conn_cfg = backend_cfg.get("connection", {})
        req_cfg = backend_cfg.get("request", {})

        api_key_env_var = auth_cfg.get("api_key_env_var", "DEEPSEEK_API_KEY")
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(
                f"[{backend_name}] Environment variable '{api_key_env_var}' is not set.\n"
                f"Please set it with: $env:{api_key_env_var} = 'your-api-key'"
            )

        base_url = conn_cfg.get("base_url", "https://api.deepseek.com/v1")
        if not base_url:
            raise ValueError(f"[{backend_name}] Missing connection.base_url")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.default_temperature = req_cfg.get("temperature", 0.2)
        self.default_top_p = req_cfg.get("top_p", 0.8)
        self.default_max_tokens = req_cfg.get("max_tokens", 4096)
        self.default_timeout = req_cfg.get(
            "timeout_seconds",
            llm_cfg.get("defaults", {}).get("timeout_seconds", 120)
        )

    def _build_messages(self, request: LLMRequest) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        messages.append({"role": "user", "content": request.prompt})
        return messages

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        # Enlever les blocs de code markdown si présents
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        return text

    def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            messages = self._build_messages(request)

            temperature = (
                request.temperature
                if request.temperature is not None
                else self.default_temperature
            )
            top_p = request.top_p if request.top_p is not None else self.default_top_p
            max_tokens = (
                request.max_tokens
                if request.max_tokens is not None
                else self.default_max_tokens
            )

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=self.default_timeout,
            )

            text = ""
            finish_reason = None
            usage = None

            if response.choices:
                finish_reason = response.choices[0].finish_reason
                if response.choices[0].message:
                    text = response.choices[0].message.content or ""

            if getattr(response, "usage", None):
                usage = response.usage.model_dump()

            text = self._clean_text(text)

            return LLMResponse(
                success=True,
                text=text,
                backend=self.backend_name,
                model=self.model_name,
                raw_response=response.model_dump(),
                error=None,
                metadata={
                    "finish_reason": finish_reason,
                    "usage": usage,
                    **request.metadata,
                },
            )

        except Exception as e:
            return LLMResponse(
                success=False,
                text="",
                backend=self.backend_name,
                model=self.model_name,
                raw_response=None,
                error=str(e),
                metadata={**request.metadata},
            )