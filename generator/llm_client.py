"""Тонкая обертка над LLM клиентом."""

from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class LLMClient:
    """Обертка над OpenAI-совместимым клиентом (VseGPT)."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Инициализация LLM клиента.
        
        Вся работа с переменными окружения находится здесь.
        
        Args:
            api_key: API ключ (если не указан, берется из VSEGPT_API_KEY)
            base_url: Базовый URL API (если не указан, берется из VSEGPT_BASE_URL или используется дефолт)
            model: Модель для генерации (если не указана, берется из GENERATION_MODEL или CLASSIFICATION_MODEL)
        """
        # API ключ из env
        self.api_key = api_key or os.getenv("VSEGPT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set VSEGPT_API_KEY in .env file."
            )
        
        # Base URL из env или дефолт
        self.base_url = base_url or os.getenv("VSEGPT_BASE_URL", "https://api.vsegpt.ru/v1")
        
        # Модель из env или дефолт
        self.model = model or os.getenv("GENERATION_MODEL") or os.getenv("CLASSIFICATION_MODEL", "anthropic/claude-3-haiku")
        
        # Инициализация OpenAI-совместимого клиента
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    async def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """
        Выполняет запрос к LLM с системным и пользовательским сообщениями.
        
        Args:
            system: Системное сообщение (роль и инструкции)
            user: Пользовательское сообщение (промпт с контекстом)
            temperature: Температура генерации (0.0-1.0)
            max_tokens: Максимальное количество токенов в ответе
        
        Returns:
            Сгенерированный текст ответа
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            raise
