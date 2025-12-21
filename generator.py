"""Shim-модуль для обратной совместимости с монолитным generator.py.

Все компоненты генератора перенесены в пакет generator/.
Этот файл оставлен только для обратной совместимости со старыми импортами.
"""

# Импортируем все из нового пакета
from generator import (
    GenerationResult,
    Citation,
    ContextItem,
    AnswerPolicy,
    PermissionPolicy,
    choose_policy,
    determine_permission_policy,
    AnswerGenerator,
    LLMClient,
    get_generator,
)

# Экспортируем все для обратной совместимости
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
    "get_generator",
]
