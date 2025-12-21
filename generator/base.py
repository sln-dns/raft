"""Базовые типы и интерфейсы для генератора."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Citation:
    """Цитата из регуляций с привязкой к источнику."""
    anchor: str
    quote: str
    chunk_id: Optional[str] = None


@dataclass
class ContextItem:
    """Элемент контекста для генерации ответа."""
    chunk_id: str
    section_number: str
    section_title: str
    text_raw: str
    anchor: Optional[str] = None
    score: Optional[float] = None  # Финальный score из ретривера
    flags: Dict[str, bool] = field(default_factory=dict)  # is_seed, is_parent, is_ref, is_sibling
    similarity: Optional[float] = None  # Для обратной совместимости
    chunk_kind: Optional[str] = None


@dataclass
class GenerationResult:
    """Структурированный результат генерации ответа."""
    answer_text: str
    citations: List[Citation] = field(default_factory=list)
    policy: str = ""  # "permitted" | "prohibited" | "conditional" | "unclear" | ""
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Обратная совместимость: можно использовать как строку."""
        return self.answer_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результат в словарь для сериализации."""
        return {
            "answer_text": self.answer_text,
            "citations": [
                {
                    "anchor": cit.anchor,
                    "quote": cit.quote,
                    "chunk_id": cit.chunk_id
                }
                for cit in self.citations
            ],
            "policy": self.policy,
            "meta": self.meta
        }


class PromptBuilder(ABC):
    """Базовый интерфейс для построителей промптов."""
    
    @abstractmethod
    def build(
        self,
        question: str,
        context_items: List[ContextItem],
        classification: Any,  # QuestionClassification
        **kwargs
    ) -> str:
        """
        Строит промпт для генерации ответа.
        
        Args:
            question: Вопрос пользователя
            context_items: Элементы контекста
            classification: Классификация вопроса
            **kwargs: Дополнительные параметры
        
        Returns:
            Промпт для LLM
        """
        pass
