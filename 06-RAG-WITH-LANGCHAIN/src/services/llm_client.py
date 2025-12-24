"""
LLM API Client
OpenAI ve Anthropic destegi ile soru uretimi
"""

import json
from abc import ABC, abstractmethod
from typing import Optional
import asyncio

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings


class BaseLLMClient(ABC):
    """LLM Client temel sinifi"""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        LLM'den yanit al
        
        Args:
            system_prompt: Sistem rolu promptu
            user_prompt: Kullanici promptu
            temperature: Randomness (0-2)
            max_tokens: Maksimum token sayisi
            
        Returns:
            LLM yaniti (string)
        """
        pass


class OpenAIClient(BaseLLMClient):
    """
    OpenAI GPT modelleri icin client
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Args:
            api_key: OpenAI API key (None ise settings'den alinir)
            model: Model adi (None ise settings'den alinir)
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        self.client = AsyncOpenAI(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """OpenAI API ile yanit al"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},  # JSON mode
        )

        return response.choices[0].message.content


class AnthropicClient(BaseLLMClient):
    """
    Anthropic Claude modelleri icin client (opsiyonel)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model

        if self.api_key:
            try:
                from anthropic import AsyncAnthropic
                self.client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                self.client = None
        else:
            self.client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Anthropic API ile yanit al"""
        if self.client is None:
            raise ValueError("Anthropic client baslatilmadi. API key kontrol edin.")

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )

        return response.content[0].text


class LLMClientFactory:
    """
    LLM Client factory pattern
    """

    _instances: dict[str, BaseLLMClient] = {}

    @classmethod
    def create(
        cls,
        provider: Optional[str] = None,
        **kwargs,
    ) -> BaseLLMClient:
        """
        LLM client olustur veya mevcut instance'i don
        
        Args:
            provider: LLM provider (openai/anthropic)
            **kwargs: Provider-spesifik argumanlar
            
        Returns:
            BaseLLMClient instance
        """
        provider = provider or settings.llm_provider

        cache_key = f"{provider}:{kwargs.get('model', 'default')}"

        if cache_key not in cls._instances:
            if provider == "openai":
                cls._instances[cache_key] = OpenAIClient(**kwargs)
            elif provider == "anthropic":
                cls._instances[cache_key] = AnthropicClient(**kwargs)
            else:
                raise ValueError(f"Desteklenmeyen provider: {provider}")

        return cls._instances[cache_key]

    @classmethod
    def clear_cache(cls):
        """Instance cache'i temizle"""
        cls._instances.clear()


def parse_llm_response(response: str) -> dict:
    """
    LLM yanitini JSON olarak parse et
    
    Args:
        response: LLM yaniti (string)
        
    Returns:
        Parsed JSON dict
        
    Raises:
        json.JSONDecodeError: Parse hatasi
    """
    response = response.strip()

    # ```json ... ``` bloklarini temizle
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end == -1:
            # Closing backticks missing, use rest of string
            response = response[start:].strip()
        else:
            response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end == -1:
            # Closing backticks missing, use rest of string
            response = response[start:].strip()
        else:
            response = response[start:end].strip()

    return json.loads(response)


# Default client instance
def get_llm_client() -> BaseLLMClient:
    """
    Default LLM client instance al
    """
    return LLMClientFactory.create()

