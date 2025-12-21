"""Политики ответов и их определение."""

from enum import Enum
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AnswerPolicy(str, Enum):
    """Политика генерации ответа."""
    STRICT_CITATION = "strict_citation"  # Только цитаты, без интерпретации
    NAVIGATION = "navigation"  # Навигационные ответы (структура документа)
    SUMMARY = "summary"  # Обзорные ответы
    QUOTED_ANSWER = "quoted_answer"  # Ответ с обязательными цитатами
    LISTING = "listing"  # Ответ в виде перечисления


class PermissionPolicy(str, Enum):
    """Политика разрешения/запрета (для permission/disclosure вопросов)."""
    PERMITTED = "permitted"
    PROHIBITED = "prohibited"
    CONDITIONAL = "conditional"
    UNCLEAR = "unclear"
    NONE = ""  # Не применимо для данного типа вопроса


def determine_permission_policy(
    category: str,
    retriever_signals: Optional[Dict[str, Any]] = None
) -> PermissionPolicy:
    """
    Определяет политику разрешения/запрета на основе категории и сигналов ретривера.
    
    Args:
        category: Категория вопроса
        retriever_signals: Сигналы от ретривера (например, policy_signal, yesno_signal)
    
    Returns:
        PermissionPolicy
    """
    # Если есть явный сигнал от ретривера
    if retriever_signals:
        policy_signal = retriever_signals.get("policy_signal", "")
        if policy_signal:
            try:
                return PermissionPolicy(policy_signal)
            except ValueError:
                pass
        
        yesno_signal = retriever_signals.get("yesno_signal", "")
        if yesno_signal == "yes":
            return PermissionPolicy.PERMITTED
        elif yesno_signal == "no":
            return PermissionPolicy.PROHIBITED
    
    # Для категорий, связанных с разрешениями/запретами
    if category in ("permission / disclosure", "procedural / best practices"):
        return PermissionPolicy.UNCLEAR  # По умолчанию unclear, если нет сигнала
    
    # Для остальных категорий политика не применима
    return PermissionPolicy.NONE


def choose_policy(
    category: str,
    classification_confidence: float,
    signals: Optional[Dict[str, Any]] = None,
    question: Optional[str] = None
) -> AnswerPolicy:
    """
    Выбирает политику генерации ответа на основе категории, уверенности и сигналов.
    
    Args:
        category: Категория вопроса
        classification_confidence: Уверенность классификации (0.0-1.0)
        signals: Сигналы от ретривера (policy_signal, yesno_signal и т.д.)
        question: Текст вопроса (для определения навигационных вопросов)
    
    Returns:
        AnswerPolicy
    """
    signals = signals or {}
    
    # Правило 1: citation-required -> STRICT_CITATION
    if category == "citation-required":
        logger.info(f"Выбран policy: STRICT_CITATION (category: {category})")
        return AnswerPolicy.STRICT_CITATION
    
    # Правило 2: навигационные вопросы -> NAVIGATION
    if question:
        question_lower = question.lower()
        navigation_keywords = ["which part", "where is", "where are", "where does", "which section", "which subpart"]
        if any(kw in question_lower for kw in navigation_keywords):
            logger.info(f"Выбран policy: NAVIGATION (навигационный вопрос)")
            return AnswerPolicy.NAVIGATION
    
    # Правило 3: overview/purpose -> SUMMARY
    if category == "overview / purpose":
        logger.info(f"Выбран policy: SUMMARY (category: {category})")
        return AnswerPolicy.SUMMARY
    
    # Правило 4: definition -> QUOTED_ANSWER (обязательная цитата)
    if category == "definition":
        logger.info(f"Выбран policy: QUOTED_ANSWER (category: {category})")
        return AnswerPolicy.QUOTED_ANSWER
    
    # Правило 4.5: regulatory_principle -> QUOTED_ANSWER (принципы требуют объяснения с цитатами)
    if category == "regulatory_principle":
        logger.info(f"Выбран policy: QUOTED_ANSWER (category: {category})")
        return AnswerPolicy.QUOTED_ANSWER
    
    # Правило 5: procedural/best practices -> QUOTED_ANSWER (Yes/No/Unclear + цитата)
    if category == "procedural / best practices":
        logger.info(f"Выбран policy: QUOTED_ANSWER (category: {category})")
        return AnswerPolicy.QUOTED_ANSWER
    
    # Правило 6: scope/applicability, penalties, permission/disclosure -> LISTING
    if category in ("scope / applicability", "penalties", "permission / disclosure"):
        logger.info(f"Выбран policy: LISTING (category: {category})")
        return AnswerPolicy.LISTING
    
    # Правило 7: other -> QUOTED_ANSWER (безопаснее чем summary)
    if category == "other":
        logger.info(f"Выбран policy: QUOTED_ANSWER (category: {category}, fallback)")
        return AnswerPolicy.QUOTED_ANSWER
    
    # Fallback: если категория не распознана, используем QUOTED_ANSWER
    logger.warning(f"Неизвестная категория '{category}', используется QUOTED_ANSWER как fallback")
    return AnswerPolicy.QUOTED_ANSWER
