"""Регистрация и выбор ретриверов по категориям."""

from typing import Optional
import logging

from .base import BaseRetriever

# Импортируем все ретриверы из модулей
from .overview_purpose import OverviewPurposeRetriever
from .navigation import NavigationRetriever
from .definition import DefinitionRetriever
from .scope import ScopeRetriever
from .penalties import PenaltiesRetriever
from .procedural import ProceduralRetriever
from .permission_disclosure import PermissionDisclosureRetriever
from .citation import CitationRetriever
from .general import GeneralRetriever

logger = logging.getLogger(__name__)


def get_retriever_for_category(category: str, question: Optional[str] = None, db_connection=None) -> BaseRetriever:
    """
    Возвращает подходящий ретривер для категории вопроса.
    
    Args:
        category: Категория вопроса
        question: Текст вопроса (для определения навигационных вопросов)
        db_connection: Подключение к базе данных (если None, создается новое)
    
    Returns:
        Экземпляр ретривера
    """
    # Маппинг категорий на классы ретриверов
    retriever_map = {
        "overview / purpose": OverviewPurposeRetriever,
        "definition": DefinitionRetriever,
        "regulatory_principle": ProceduralRetriever,  # Используем ProceduralRetriever для принципов
        "scope / applicability": ScopeRetriever,
        "penalties": PenaltiesRetriever,
        "procedural / best practices": ProceduralRetriever,
        "permission / disclosure": PermissionDisclosureRetriever,
        "citation-required": CitationRetriever,
    }
    
    # Для навигационных вопросов (которые спрашивают "which part", "where is") используем NavigationRetriever
    # Определяем по ключевым словам в вопросе
    if question:
        question_lower = question.lower()
        navigation_keywords = ["which part", "where is", "where are", "where does", "which section", "which subpart"]
        if any(kw in question_lower for kw in navigation_keywords):
            logger.info("Using NavigationRetriever (retrievers.navigation)")
            return NavigationRetriever(db_connection)
    
    retriever_class = retriever_map.get(category, GeneralRetriever)  # GeneralRetriever как fallback
    
    logger.info(f"Using {retriever_class.__name__} for category '{category}'")
    
    return retriever_class(db_connection)
