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
    ] = Field(..., description="Категория вопроса (смысл вопроса)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность классификации (0.0-1.0)")
    reasoning: str = Field(..., description="Краткое обоснование классификации")
    
    # Шаг 1: Новые поля для цитирования (с дефолтами для обратной совместимости)
    require_citations: bool = Field(
        default=False,
        description="Требуются ли цитаты в ответе (true если вопрос явно просит цитировать)"
    )
    citation_mode: Literal["none", "quoted", "strict"] = Field(
        default="none",
        description="Режим цитирования: none (не нужны), quoted (с цитатами в тексте), strict (строгое цитирование без пересказа)"
    )
    anchor_hint: Optional[str] = Field(
        default=None,
        description="Подсказка по anchor (например, '§164.512(f)' для law enforcement вопросов)"
    )
    scope_hint: Optional[str] = Field(
        default=None,
        description="Подсказка по scope (например, 'law enforcement', 'family disclosure', 'minimum necessary')"
    )


class QuestionClassifier:
    """Классификатор вопросов для HIPAA регуляций."""
    
    # Описания категорий для промпта (на английском)
    CATEGORY_DESCRIPTIONS = {
        "overview / purpose": "General overview, purpose of regulations, which parts cover what",
        "definition": "Term definitions (what does X mean) - simple dictionary terms",
        "regulatory_principle": "Regulatory principles and concepts (minimum necessary, reasonable safeguards, addressable implementation specification, etc.) - normative principles requiring explanation of application context",
        "scope / applicability": "Scope of applicability, which entities are covered by regulations",
        "penalties": "Civil penalties, sanctions",
        "procedural / best practices": "Procedures, best practices, encryption, safeguards",
        "permission / disclosure": "Permissions for information disclosure, whether disclosure is allowed",
        "conditional / dependency": "Conditions and dependencies (if X, which sections apply)",
        "citation-required": "Citation of specific regulation texts is required",
        "other": "Other questions not fitting the above categories"
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
        """Builds system prompt for classification (in English)."""
        categories_text = "\n".join([
            f"- **{cat}**: {desc}"
            for cat, desc in self.CATEGORY_DESCRIPTIONS.items()
        ])
        
        return f"""You are an expert at classifying questions about HIPAA regulations.

Your task is to classify the user's question into one of the following categories:

{categories_text}

Return classification in JSON format with fields:
- category: one of the categories above (the MEANING of the question)
- confidence: number from 0.0 to 1.0 (classification confidence)
- reasoning: brief justification (1-2 sentences)
- require_citations: boolean (true if question explicitly asks to cite, e.g., "cite", "quote", "show the text")
- citation_mode: "none" | "quoted" | "strict"
  * "none" - citations not needed (regular questions)
  * "quoted" - citations needed in answer text (questions like "what does X mean" require exact definitions)
  * "strict" - strict citation without interpretation (questions like "cite the exact text", "show the regulation")
- anchor_hint: string | null (anchor hint, e.g., "§164.512(f)" for law enforcement, "§160.103" for definitions)
- scope_hint: string | null (topic/scope hint, e.g., "law enforcement", "family disclosure", "minimum necessary")

CRITICAL RULES:
1. category describes the MEANING of the question, NOT the citation mode
2. If question contains "cite", "quote", "show the text", "exact text", "verbatim" -> require_citations=true
3. If question contains explicit anchor (e.g., "§164.512(f)", "§160.103") -> set anchor_hint to that anchor (normalize spaces: "§ 164.512(f)" -> "§164.512(f)")
4. If question asks for definition of a term -> require_citations=true, citation_mode="quoted"
5. If question contains "cite" + strict topic (law enforcement, suspect, fugitive, victim, missing person) -> citation_mode="strict", category="citation-required"
6. If question contains "cite" but NO strict topic -> citation_mode="quoted", category stays as determined by question meaning
7. If question asks "what does X mean" and X is a regulatory principle (minimum necessary, reasonable safeguards, etc.) -> category="regulatory_principle", citation_mode="none" (unless "cite" is present, then "quoted")
8. For regulatory_principle questions, citation_mode is usually "none" or "quoted" (if "cite" present), NOT "strict" unless explicit anchor is given
9. If question mentions law enforcement, police, court, warrant, subpoena -> scope_hint="law enforcement", anchor_hint may be "§164.512(f)"
10. anchor_hint and scope_hint: only fill if you can confidently determine from the question

FEW-SHOT EXAMPLES:

Example 1:
Question: "Cite §164.512(f)(2) regarding disclosures to identify suspects."
Response:
{{
  "category": "citation-required",
  "confidence": 0.95,
  "reasoning": "Question explicitly requests citation of a specific regulation section with anchor",
  "require_citations": true,
  "citation_mode": "strict",
  "anchor_hint": "§164.512(f)(2)",
  "scope_hint": "suspect/fugitive/witness/missing person"
}}

Example 2:
Question: "How long retain documentation? Cite."
Response:
{{
  "category": "other",
  "confidence": 0.85,
  "reasoning": "Question about documentation retention with citation request, but no specific regulation section mentioned",
  "require_citations": true,
  "citation_mode": "quoted",
  "anchor_hint": null,
  "scope_hint": "cite_requested"
}}

Example 3:
Question: "Define business associate. Cite."
Response:
{{
  "category": "definition",
  "confidence": 0.9,
  "reasoning": "Question asks for definition of a term with citation request",
  "require_citations": true,
  "citation_mode": "quoted",
  "anchor_hint": null,
  "scope_hint": "cite_requested"
}}

Example 4:
Question: "What does minimum necessary mean?"
Response:
{{
  "category": "regulatory_principle",
  "confidence": 0.9,
  "reasoning": "Question asks about a regulatory principle/concept, not a simple dictionary definition",
  "require_citations": false,
  "citation_mode": "none",
  "anchor_hint": null,
  "scope_hint": "minimum necessary"
}}

Example 5:
Question: "What does minimum necessary mean? Cite."
Response:
{{
  "category": "regulatory_principle",
  "confidence": 0.9,
  "reasoning": "Question asks about a regulatory principle with citation request",
  "require_citations": true,
  "citation_mode": "quoted",
  "anchor_hint": null,
  "scope_hint": "minimum necessary"
}}"""

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
            {"role": "user", "content": f"Classify the following question:\n\n{question}"}
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
            messages[-1]["content"] += "\n\nReturn response ONLY in JSON format with fields: category, confidence, reasoning, require_citations, citation_mode, anchor_hint, scope_hint"
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
