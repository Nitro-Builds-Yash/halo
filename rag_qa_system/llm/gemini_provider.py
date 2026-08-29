from __future__ import annotations
import json
import re
from typing import Dict, Any, Optional
from .base_provider import BaseLLMProvider
from rag_qa_system.core.config import get_settings


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.gemini_api_key
        self.model = model
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            from google import genai
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
        except Exception:
            self.client = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        if not self.client:
            from .mock_provider import MockLLMProvider
            return MockLLMProvider().generate(prompt, system_prompt, temperature)

        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
            )
            return response.text or ""
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
