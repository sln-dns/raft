"""Построитель контекста из чанков для генерации."""

from typing import List, Dict, Any, Optional
from .base import ContextItem
from .policy import AnswerPolicy


def build_context(
    chunks: List[Dict[str, Any]],
    policy: AnswerPolicy
) -> List[ContextItem]:
    """
    Строит нормализованный контекст из чанков с учетом политики генерации.
    
    Args:
        chunks: Список чанков из ретривера
        policy: Политика генерации ответа (определяет лимиты)
    
    Returns:
        Список ContextItem, отсортированный и ограниченный по размеру
    """
    if not chunks:
        return []
    
    # Преобразуем чанки в ContextItem
    context_items = []
    for chunk in chunks:
        # Извлекаем scores
        scores = chunk.get("scores", {})
        if isinstance(scores, dict):
            final_score = scores.get("final_score")
        else:
            final_score = None
        
        # Извлекаем flags из chunk (если есть)
        flags = {}
        if chunk.get("is_seed"):
            flags["is_seed"] = True
        if chunk.get("is_parent"):
            flags["is_parent"] = True
        if chunk.get("is_ref"):
            flags["is_ref"] = True
        if chunk.get("is_sibling"):
            flags["is_sibling"] = True
        
        item = ContextItem(
            chunk_id=chunk.get("chunk_id", ""),
            section_number=chunk.get("section_number", "N/A"),
            section_title=chunk.get("section_title", "N/A"),
            text_raw=chunk.get("text_raw", ""),
            anchor=chunk.get("anchor"),
            score=final_score,
            flags=flags,
            similarity=final_score,  # Для обратной совместимости
            chunk_kind=chunk.get("chunk_kind")
        )
        context_items.append(item)
    
    # Сортировка: сначала по section_id, затем по anchor
    # Используем section_number как proxy для section_id (они обычно совпадают)
    def sort_key(item: ContextItem) -> tuple:
        section_id = item.section_number or ""
        anchor = item.anchor or ""
        return (section_id, anchor)
    
    context_items.sort(key=sort_key)
    
    # Ограничение размера в зависимости от политики
    max_items = _get_max_items_for_policy(policy)
    if len(context_items) > max_items:
        context_items = context_items[:max_items]
    
    return context_items


def _get_max_items_for_policy(policy: AnswerPolicy) -> int:
    """
    Возвращает максимальное количество элементов контекста для политики.
    
    Args:
        policy: Политика генерации ответа
    
    Returns:
        Максимальное количество элементов
    """
    limits = {
        AnswerPolicy.STRICT_CITATION: 10,  # 6-10
        AnswerPolicy.SUMMARY: 2,  # 1-2 крупных
        AnswerPolicy.LISTING: 10,  # 6-10
        AnswerPolicy.QUOTED_ANSWER: 6,  # 4-6
        AnswerPolicy.NAVIGATION: 10,  # Для навигации можно больше
    }
    return limits.get(policy, 6)  # По умолчанию 6


def build_context_items(chunks: List[Dict[str, Any]]) -> List[ContextItem]:
    """
    Преобразует список чанков в структурированные элементы контекста.
    
    Устаревшая функция для обратной совместимости.
    Используйте build_context() с указанием policy.
    
    Args:
        chunks: Список чанков из ретривера
    
    Returns:
        Список ContextItem
    """
    # Используем QUOTED_ANSWER как дефолтную политику
    return build_context(chunks, AnswerPolicy.QUOTED_ANSWER)
