"""Валидация citations из LLM ответов."""

from typing import List, Dict, Any, Optional, Tuple
import re
import json
import logging

from .base import ContextItem, Citation

logger = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    """
    Нормализует пробелы в тексте для сравнения.
    
    Args:
        text: Исходный текст
    
    Returns:
        Текст с нормализованными пробелами
    """
    # Заменяем множественные пробелы на один
    text = re.sub(r'\s+', ' ', text)
    # Убираем пробелы в начале и конце
    return text.strip()


def extract_relevant_quote(text_raw: str, max_length: int = 300) -> str:
    """
    Извлекает релевантный фрагмент из text_raw для использования как quote.
    
    Приоритет:
    1. Первое предложение (до точки, если есть)
    2. Первые max_length символов
    
    Args:
        text_raw: Исходный текст
        max_length: Максимальная длина фрагмента
    
    Returns:
        Релевантный фрагмент текста
    """
    if not text_raw:
        return ""
    
    # Пробуем найти первое предложение (до точки, восклицательного или вопросительного знака)
    sentence_end = re.search(r'[.!?]\s+', text_raw)
    if sentence_end:
        first_sentence = text_raw[:sentence_end.end()].strip()
        # Если предложение не слишком длинное, используем его
        if len(first_sentence) <= max_length:
            return first_sentence
    
    # Иначе берем первые max_length символов, обрезая по слову
    if len(text_raw) <= max_length:
        return text_raw.strip()
    
    # Обрезаем по последнему пробелу перед max_length
    truncated = text_raw[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.7:  # Если пробел не слишком далеко от конца
        return truncated[:last_space].strip() + "..."
    
    return truncated.strip() + "..."


def validate_citation(
    citation: Dict[str, str],
    context_items: List[ContextItem],
    auto_fix_quote: bool = True
) -> Optional[Citation]:
    """
    Валидирует одну citation из LLM ответа.
    
    Если anchor валиден, но quote не найден в text_raw, и auto_fix_quote=True,
    автоматически заменяет quote на релевантный фрагмент из text_raw.
    
    Args:
        citation: Словарь с anchor и quote из LLM
        context_items: Элементы контекста для проверки
        auto_fix_quote: Если True, автоматически исправляет quote при валидном anchor
    
    Returns:
        Валидный Citation или None, если валидация не прошла
    """
    anchor = citation.get("anchor", "").strip()
    quote = citation.get("quote", "").strip()
    
    if not anchor:
        logger.warning(f"Citation missing anchor: {citation}")
        return None
    
    # Ищем соответствующий context item по anchor
    matching_item = None
    for item in context_items:
        if item.anchor and item.anchor.strip() == anchor:
            matching_item = item
            break
    
    if not matching_item:
        logger.warning(f"Anchor '{anchor}' not found in context items")
        return None
    
    # Если quote отсутствует, но auto_fix_quote включен - извлекаем релевантный фрагмент
    if not quote:
        if auto_fix_quote:
            quote = extract_relevant_quote(matching_item.text_raw)
            logger.info(f"Auto-fixed missing quote for anchor {anchor}: extracted {len(quote)} chars")
        else:
            logger.warning(f"Citation missing quote for anchor {anchor}")
            return None
    
    # Нормализуем whitespace для сравнения
    normalized_quote = normalize_whitespace(quote)
    normalized_text_raw = normalize_whitespace(matching_item.text_raw)
    
    # Проверяем, что quote является подстрокой text_raw
    if normalized_quote.lower() not in normalized_text_raw.lower():
        if auto_fix_quote:
            # Автоматически исправляем quote, используя релевантный фрагмент из text_raw
            fixed_quote = extract_relevant_quote(matching_item.text_raw)
            logger.info(
                f"Auto-fixed invalid quote for anchor {anchor}: "
                f"original quote length {len(quote)}, fixed quote length {len(fixed_quote)}"
            )
            quote = fixed_quote
        else:
            logger.warning(
                f"Quote not found in context item text_raw. "
                f"Anchor: {anchor}, Quote length: {len(quote)}, Text length: {len(matching_item.text_raw)}"
            )
            return None
    
    # Возвращаем валидную citation
    return Citation(
        anchor=anchor,
        quote=quote,
        chunk_id=matching_item.chunk_id
    )


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Извлекает JSON из текста, который может содержать markdown или другие обертки.
    
    Args:
        text: Текст, который может содержать JSON
    
    Returns:
        Распарсенный JSON или None
    """
    # Пробуем найти JSON блок в markdown code fence
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Пробуем найти JSON объект напрямую
    # Ищем первую открывающую скобку и последнюю закрывающую
    start_idx = text.find('{')
    if start_idx != -1:
        # Находим соответствующую закрывающую скобку
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > start_idx:
            try:
                return json.loads(text[start_idx:end_idx])
            except json.JSONDecodeError:
                pass
    
    # Последняя попытка - весь текст как JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def parse_and_validate_citations(
    llm_response: str,
    context_items: List[ContextItem],
    require_citations: bool = False,
    auto_fix_quote: bool = True
) -> tuple[str, List[Citation]]:
    """
    Парсит JSON ответ от LLM и валидирует citations.
    
    Если anchor валиден, но quote не найден, автоматически исправляет quote
    (если auto_fix_quote=True), используя релевантный фрагмент из text_raw.
    
    Args:
        llm_response: Ответ от LLM (должен быть JSON)
        context_items: Элементы контекста для валидации
        require_citations: Если True и нет валидных citations - возвращает ошибку
        auto_fix_quote: Если True, автоматически исправляет quote при валидном anchor
    
    Returns:
        Tuple (answer_text, valid_citations)
    """
    # Извлекаем JSON из ответа
    json_data = extract_json_from_text(llm_response)
    
    if json_data is None:
        logger.warning("LLM response is not valid JSON, treating as plain text")
        if require_citations:
            return "Insufficient context to provide exact citation.", []
        return llm_response.strip(), []
    
    # Извлекаем answer
    answer_text = json_data.get("answer", "").strip()
    if not answer_text:
        answer_text = llm_response.strip()  # Fallback на весь ответ
    
    # Извлекаем и валидируем citations
    citations_data = json_data.get("citations", [])
    if not isinstance(citations_data, list):
        citations_data = []
    
    valid_citations = []
    auto_fixed_count = 0
    
    for citation_data in citations_data:
        if not isinstance(citation_data, dict):
            continue
        
        # Проверяем, был ли anchor валиден до валидации
        anchor = citation_data.get("anchor", "").strip()
        original_quote = citation_data.get("quote", "").strip()
        
        valid_citation = validate_citation(
            citation_data, 
            context_items, 
            auto_fix_quote=auto_fix_quote
        )
        
        if valid_citation:
            # Проверяем, был ли quote исправлен
            if auto_fix_quote and original_quote and valid_citation.quote != original_quote:
                auto_fixed_count += 1
            valid_citations.append(valid_citation)
        else:
            logger.debug(f"Invalid citation filtered out: {citation_data}")
    
    # Если citations обязательны и их нет - возвращаем ошибку
    if require_citations and not valid_citations:
        logger.warning("Required citations not found or invalid after validation")
        return "Insufficient context to provide exact citation.", []
    
    if auto_fixed_count > 0:
        logger.info(f"Auto-fixed {auto_fixed_count} citation(s) with invalid quotes")
    
    logger.info(f"Validated {len(valid_citations)} citations from {len(citations_data)} provided")
    
    return answer_text, valid_citations
