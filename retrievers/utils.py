"""Утилиты для ретриверов."""

from typing import List, Optional, Dict
from pathlib import Path
import psycopg
import re

from .base import CandidateChunk


def get_db_connection():
    """Создает подключение к базе данных."""
    user = Path.home().name
    return psycopg.connect(
        host="localhost",
        dbname="raft",
        user=user,
    )


def normalize_fts_scores(candidates: List[CandidateChunk]) -> None:
    """
    Нормализует FTS scores в списке кандидатов.
    
    Модифицирует candidates in-place, нормализуя fts_score на максимальное значение.
    """
    if not candidates:
        return
    
    max_fts = max(c.fts_score for c in candidates if c.fts_score > 0)
    if max_fts > 0:
        for candidate in candidates:
            if candidate.fts_score > 0:
                candidate.fts_score = candidate.fts_score / max_fts


def merge_candidates(
    vector_candidates: List[CandidateChunk],
    fts_candidates: List[CandidateChunk]
) -> Dict[str, CandidateChunk]:
    """
    Объединяет vector и FTS кандидатов в словарь по chunk_id.
    
    Returns:
        Словарь {chunk_id: CandidateChunk} с объединенными оценками
    """
    merged_dict: Dict[str, CandidateChunk] = {}
    
    # Добавляем vector кандидатов
    for candidate in vector_candidates:
        merged_dict[candidate.chunk_id] = candidate
    
    # Объединяем с FTS кандидатами
    for candidate in fts_candidates:
        if candidate.chunk_id in merged_dict:
            # Объединяем оценки
            existing = merged_dict[candidate.chunk_id]
            existing.fts_score = candidate.fts_score
        else:
            # Добавляем новый
            merged_dict[candidate.chunk_id] = candidate
    
    return merged_dict


def calculate_final_score(
    candidate: CandidateChunk,
    vector_weight: float,
    fts_weight: float,
    additional_score: float = 0.0,
    additional_weight: float = 0.0
) -> float:
    """
    Вычисляет финальный score для кандидата.
    
    Args:
        candidate: Кандидат с vector_score и fts_score
        vector_weight: Вес vector score
        fts_weight: Вес FTS score
        additional_score: Дополнительный score (например, keyword_score, amount_score)
        additional_weight: Вес дополнительного score
    
    Returns:
        Финальный score
    """
    base_score = vector_weight * candidate.vector_score + fts_weight * candidate.fts_score
    
    if additional_weight > 0 and additional_score > 0:
        return (1.0 - additional_weight) * base_score + additional_weight * additional_score
    
    return base_score


def calculate_amount_score(text: str) -> float:
    """
    Вычисляет amount_score по regex на текст.
    
    Returns:
        1.0 если найден паттерн суммы ($, USD, числа с разделителями)
        0.5 если просто есть числа
        0.0 если нет чисел
    """
    text_lower = text.lower()
    
    # Проверка на наличие $ или USD
    if re.search(r'\$|\busd\b', text_lower):
        return 1.0
    
    # Проверка на числа с разделителями (например, 1,000,000)
    if re.search(r'\d{1,3}(,\d{3})+', text):
        return 1.0
    
    # Проверка на ключевые слова
    amount_keywords = ["dollar", "amount", "maximum", "minimum", "per violation", "per year", "tier"]
    if any(kw in text_lower for kw in amount_keywords):
        # Если есть ключевые слова + числа
        if re.search(r'\d+', text):
            return 1.0
    
    # Проверка на просто числа
    if re.search(r'\d+', text):
        return 0.5
    
    return 0.0


def calculate_keyword_score_procedural(text: str, topic_tokens: List[str]) -> float:
    """
    Вычисляет keyword_score для procedural/best practices ретривера.
    
    Args:
        text: Текст для анализа
        topic_tokens: Список токенов темы (например, encryption tokens)
    
    Returns:
        keyword_score в диапазоне 0..1
    """
    text_lower = text.lower()
    score = 0.0
    
    # Проверка на тему (например, encryption)
    encryption_tokens = ["encrypt", "encryption", "decrypt", "cryptograph"]
    if any(token in text_lower for token in encryption_tokens):
        if any(token in topic_tokens for token in encryption_tokens):
            score += 1.0
    
    # Проверка на implementation specification или addressable
    if re.search(r'\b(implementation specification|addressable)\b', text_lower):
        score += 0.6
    
    # Проверка на safeguard(s) или security
    if re.search(r'\b(safeguard|safeguards|security)\b', text_lower):
        score += 0.4
    
    # Проверка на модальные "must/required/shall"
    if re.search(r'\b(must|required|shall)\b', text_lower):
        score += 0.4
    
    # Штраф за "not required" / "no requirement"
    if re.search(r'\b(not required|no requirement)\b', text_lower):
        score -= 0.3
    
    # Нормализация в диапазон 0..1
    return max(0.0, min(1.0, score))


def calculate_clause_score_disclosure(text: str, topic_tokens: List[str]) -> float:
    """
    Вычисляет clause_score для disclosure ретривера.
    
    Args:
        text: Текст для анализа
        topic_tokens: Список токенов темы (например, family, law enforcement)
    
    Returns:
        clause_score в диапазоне 0..1
    """
    text_lower = text.lower()
    score = 0.0
    
    # Модальные/разрешительные
    if re.search(r'\b(may|is permitted)\b', text_lower):
        score += 0.4
    if re.search(r'\b(may disclose|may use or disclose)\b', text_lower):
        score += 0.4
    if re.search(r'\b(without authorization)\b', text_lower):
        score += 0.2
    
    # Ограничения/исключения
    if re.search(r'\b(except|except as provided)\b', text_lower):
        score += 0.4
    if re.search(r'\b(subject to|provided that|only if)\b', text_lower):
        score += 0.4
    if re.search(r'\b(minimum necessary)\b', text_lower):
        score += 0.4
    
    # Запрещающее (для policy_signal)
    if re.search(r'\b(may not|prohibited)\b', text_lower):
        score += 0.3
    
    # Topic boost
    if topic_tokens and any(token in text_lower for token in topic_tokens):
        score += 0.6
    
    # Нормализация в диапазон 0..1
    return max(0.0, min(1.0, score))


def determine_part_hint(question: str) -> Optional[int]:
    """
    Определяет part hint из вопроса простым правилом.
    
    Args:
        question: Текст вопроса
    
    Returns:
        part hint (164, 162) или None
    """
    if not question:
        return None
    
    question_lower = question.lower()
    
    # Privacy, disclosure, PHI -> 164
    if any(term in question_lower for term in ["privacy", "disclosure", "phi", "protected health"]):
        return 164
    
    # Transactions, code set -> 162
    if any(term in question_lower for term in ["transaction", "code set", "identifier"]):
        return 162
    
    return None


def build_fts_query(question: str, boost_words: List[str], topic_tokens: Optional[List[str]] = None) -> str:
    """
    Формирует усиленный FTS query из вопроса + буст-слова + опциональные токены темы.
    
    Args:
        question: Оригинальный вопрос
        boost_words: Список буст-слов для добавления
        topic_tokens: Опциональные токены темы
    
    Returns:
        Усиленный FTS query
    """
    if not question:
        all_tokens = boost_words + (topic_tokens or [])
        return " ".join(all_tokens)
    
    all_tokens = boost_words + (topic_tokens or [])
    boost_text = " ".join(all_tokens)
    return f"{question} {boost_text}"
