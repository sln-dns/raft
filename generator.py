"""Генератор ответов на основе найденных чанков."""

from typing import List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

from classifier import QuestionClassification

load_dotenv()
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Генератор ответов с использованием LLM."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.vsegpt.ru/v1",
        model: Optional[str] = None,
    ):
        """
        Инициализация генератора ответов.
        
        Args:
            api_key: API ключ VseGPT (если не указан, берется из VSEGPT_API_KEY)
            base_url: Базовый URL API (по умолчанию VseGPT)
            model: Модель для генерации (если не указана, берется из GENERATION_MODEL или используется классификатор)
        """
        self.api_key = api_key or os.getenv("VSEGPT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API ключ не найден. Укажите VSEGPT_API_KEY в .env файле."
            )
        
        self.base_url = base_url
        # Модель для генерации (можно использовать отдельную от классификатора)
        self.model = model or os.getenv("GENERATION_MODEL") or os.getenv("CLASSIFICATION_MODEL", "anthropic/claude-3-haiku")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def _build_prompt(
        self,
        question: str,
        chunks: List[dict],
        classification: QuestionClassification
    ) -> str:
        """
        Строит промпт для генерации ответа.
        
        Args:
            question: Вопрос пользователя
            chunks: Найденные релевантные чанки
            classification: Классификация вопроса
        
        Returns:
            Промпт для LLM
        """
        # Формируем контекст из чанков
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            section_info = f"Секция {chunk.get('section_number', 'N/A')}: {chunk.get('section_title', 'N/A')}"
            if chunk.get('anchor'):
                section_info += f" ({chunk['anchor']})"
            
            context_parts.append(
                f"[{i}] {section_info}\n{chunk.get('text_raw', '')}"
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Ты - эксперт по HIPAA регуляциям. Ответь на вопрос пользователя на основе предоставленного контекста из официальных регуляций.

Категория вопроса: {classification.category}
Уверенность классификации: {classification.confidence:.0%}

Контекст из регуляций:
{context}

Вопрос пользователя: {question}

Инструкции:
1. Ответь точно на основе предоставленного контекста
2. Если информация не найдена в контексте, скажи об этом честно
3. Используй точные формулировки из регуляций, когда это важно
4. Будь конкретным и точным
5. Если нужно, укажи номер секции или anchor для ссылки

Ответ:"""
        
        return prompt
    
    async def generate(
        self,
        question: str,
        chunks: List[dict],
        classification: QuestionClassification
    ) -> str:
        """
        Генерирует ответ на основе найденных чанков.
        
        Args:
            question: Вопрос пользователя
            chunks: Список найденных релевантных чанков
            classification: Классификация вопроса
        
        Returns:
            Сгенерированный ответ
        """
        if not chunks:
            return "Извините, не удалось найти релевантную информацию для ответа на ваш вопрос в базе регуляций HIPAA."
        
        prompt = self._build_prompt(question, chunks, classification)
        
        messages = [
            {"role": "system", "content": "Ты - эксперт по HIPAA регуляциям. Отвечай точно и профессионально на основе предоставленного контекста."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Низкая температура для более точных ответов
                max_tokens=2000,
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}", exc_info=True)
            raise


# Глобальный экземпляр генератора
_generator: Optional[AnswerGenerator] = None


def get_generator() -> AnswerGenerator:
    """
    Возвращает глобальный экземпляр генератора ответов.
    
    Returns:
        Экземпляр AnswerGenerator
    """
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator
