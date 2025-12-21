"""Классификатор вопросов пользователя для HIPAA регуляций."""

import os
import json
from typing import Literal, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class QuestionClassification(BaseModel):
    """Структурированный ответ классификатора."""
    category: Literal[
        "overview / purpose",
        "definition",
        "regulatory_principle",
        "scope / applicability",
        "penalties",
        "procedural / best practices",
        "permission / disclosure",
        "conditional / dependency",
        "citation-required",
        "other"
    ] = Field(..., description="Категория вопроса")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность классификации (0.0-1.0)")
    reasoning: str = Field(..., description="Краткое обоснование классификации")


class QuestionClassifier:
    """Классификатор вопросов для HIPAA регуляций."""
    
    # Описания категорий для промпта
    CATEGORY_DESCRIPTIONS = {
        "overview / purpose": "Общий обзор, цель регуляций, какие части покрывают",
        "definition": "Определения терминов (что означает X) - простые термины из словаря",
        "regulatory_principle": "Регуляторные принципы и концепции (minimum necessary, reasonable safeguards, addressable implementation specification и т.д.) - нормативные принципы, требующие объяснения контекста применения",
        "scope / applicability": "Область применения, какие сущности подпадают под регуляции",
        "penalties": "Гражданские штрафы, наказания",
        "procedural / best practices": "Процедуры, лучшие практики, шифрование, меры защиты",
        "permission / disclosure": "Разрешения на раскрытие информации, можно ли раскрыть",
        "conditional / dependency": "Условия и зависимости (если X, какие секции применяются)",
        "citation-required": "Требуется цитирование конкретных текстов регуляций",
        "other": "Другие вопросы, не подходящие под вышеперечисленные категории"
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.vsegpt.ru/v1",
        model: Optional[str] = None,
    ):
        """
        Инициализация классификатора.
        
        Args:
            api_key: API ключ VseGPT (если не указан, берется из VSEGPT_API_KEY)
            base_url: Базовый URL API (по умолчанию VseGPT)
            model: Модель для классификации (если не указана, берется из CLASSIFICATION_MODEL)
        """
        self.api_key = api_key or os.getenv("VSEGPT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API ключ не найден. Укажите VSEGPT_API_KEY в .env файле или передайте api_key напрямую."
            )
        
        self.base_url = base_url
        self.model = model or os.getenv("CLASSIFICATION_MODEL", "anthropic/claude-3-haiku")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def _build_system_prompt(self) -> str:
        """Строит системный промпт для классификации."""
        categories_text = "\n".join([
            f"- **{cat}**: {desc}"
            for cat, desc in self.CATEGORY_DESCRIPTIONS.items()
        ])
        
        return f"""Ты - эксперт по классификации вопросов о HIPAA регуляциях.

Твоя задача - классифицировать вопрос пользователя в одну из следующих категорий:

{categories_text}

Верни классификацию в формате JSON с полями:
- category: одна из категорий выше
- confidence: число от 0.0 до 1.0 (уверенность в классификации)
- reasoning: краткое обоснование (1-2 предложения)"""

    def classify(self, question: str) -> QuestionClassification:
        """
        Классифицирует вопрос пользователя.
        
        Args:
            question: Вопрос пользователя
        
        Returns:
            QuestionClassification с категорией, уверенностью и обоснованием
        """
        system_prompt = self._build_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Классифицируй следующий вопрос:\n\n{question}"}
        ]
        
        # Используем structured output через response_format
        # Если API не поддерживает json_schema, используем обычный запрос с инструкцией вернуть JSON
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Низкая температура для более детерминированной классификации
                max_tokens=500,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "question_classification",
                        "strict": True,
                        "schema": QuestionClassification.model_json_schema()
                    }
                }
            )
        except Exception as e:
            # Fallback: если structured output не поддерживается, добавляем инструкцию в промпт
            messages[-1]["content"] += "\n\nВерни ответ ТОЛЬКО в формате JSON с полями: category, confidence, reasoning"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
        
        # Парсим ответ
        content = response.choices[0].message.content.strip()
        
        # Убираем markdown code blocks если есть
        if content.startswith("```"):
            import re
            content = re.sub(r'^```(?:json)?\s*\n', '', content)
            content = re.sub(r'\n```\s*$', '', content)
        
        try:
            data = json.loads(content)
            return QuestionClassification(**data)
        except json.JSONDecodeError as e:
            # Fallback: пытаемся извлечь JSON из текста
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return QuestionClassification(**data)
                except:
                    pass
            raise ValueError(f"Не удалось распарсить ответ модели: {e}\nОтвет: {content}")


# Глобальный экземпляр классификатора
_classifier: Optional[QuestionClassifier] = None


def get_classifier() -> QuestionClassifier:
    """
    Возвращает глобальный экземпляр классификатора.
    
    Returns:
        Экземпляр QuestionClassifier
    """
    global _classifier
    if _classifier is None:
        _classifier = QuestionClassifier()
    return _classifier


if __name__ == "__main__":
    # Примеры использования
    classifier = QuestionClassifier()
    
    test_questions = [
        "What is the overall purpose of HIPAA regulations?",
        "What does 'business associate' mean?",
        "Which entities are covered by HIPAA?",
        "What are the penalties for violating HIPAA?",
        "How should I encrypt patient data?",
        "Can I disclose patient information to a family member?",
        "If I'm a covered entity, which sections apply to me?",
        "Cite the exact text of section 160.103",
        "Tell me about HIPAA in general",
    ]
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ КЛАССИФИКАТОРА ВОПРОСОВ")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\n❓ Вопрос: {question}")
        try:
            result = classifier.classify(question)
            print(f"📋 Категория: {result.category}")
            print(f"🎯 Уверенность: {result.confidence:.2%}")
            print(f"💭 Обоснование: {result.reasoning}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
