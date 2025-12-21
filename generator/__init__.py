"""Пакет генератора ответов."""

from typing import Optional

from .base import GenerationResult, Citation, ContextItem
from .policy import AnswerPolicy, PermissionPolicy, choose_policy, determine_permission_policy
from .generator import AnswerGenerator
from .llm_client import LLMClient
from .citation_validator import validate_citation, parse_and_validate_citations

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


__all__ = [
    "GenerationResult",
    "Citation",
    "ContextItem",
    "AnswerPolicy",
    "PermissionPolicy",
    "choose_policy",
    "determine_permission_policy",
    "AnswerGenerator",
    "LLMClient",
    "validate_citation",
    "parse_and_validate_citations",
    "get_generator",
]
