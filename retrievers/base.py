"""Базовые классы и типы для ретриверов."""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CandidateChunk:
    """Кандидат-чанк с оценками."""
    chunk_id: str
    anchor: Optional[str]
    section_id: str
    section_number: str
    section_title: str
    text_raw: str
    page_start: Optional[int]
    page_end: Optional[int]
    vector_score: float
    fts_score: float
    final_score: float


@dataclass
class NavigationHit:
    """Результат навигационного поиска - структурный элемент документа."""
    part: Optional[int]
    subpart: Optional[str]
    section_id: str
    section_number: str
    section_title: str
    anchor: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    scores: Dict[str, float]  # rule_score, title_score, final_score
    explanation: str


class BaseRetriever(ABC):
    """Базовый класс для ретриверов."""
    
    @abstractmethod
    async def retrieve(
        self,
        question_embedding: List[float],
        max_results: int = 5,
        **kwargs
    ) -> List[dict]:
        """
        Поиск релевантных чанков.
        
        Args:
            question_embedding: Эмбеддинг вопроса
            max_results: Максимальное количество результатов
            **kwargs: Дополнительные параметры для конкретного ретривера
        
        Returns:
            Список словарей с информацией о чанках
        """
        pass
