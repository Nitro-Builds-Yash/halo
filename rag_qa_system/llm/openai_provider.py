from __future__ import annotations
import json
import re
from typing import Dict, Any, Optional
from .base_provider import BaseLLMProvider
from rag_qa_system.core.config import get_settings


class OpenAIProvider(BaseLLMProvider):
    """OpenAI / OpenRouter API provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.openai_api_key
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            from .mock_provider import MockLLMProvider
            return MockLLMProvider().generate(prompt, system_prompt, temperature)

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        raw = self.generate(f"{prompt}\n\nRespond strictly in valid JSON.", system_prompt)
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(raw)
        except Exception:
            from .mock_provider import MockLLMProvider
            return MockLLMProvider().generate_json(prompt, system_prompt)
