"""Post-classification override для применения доменных правил.

После классификации вопросов LLM применяются rule-based правила для переопределения категорий
на основе паттернов вопросов и словарей терминов.
"""

import re
import logging
from typing import Optional

from classifier import QuestionClassification

logger = logging.getLogger(__name__)


# Словарь регуляторных концепций/принципов (не простые определения)
REGULATORY_CONCEPTS = {
    "minimum necessary",
    "reasonable safeguards",
    "addressable implementation specification",
    "administrative safeguards",
    "technical safeguards",
    "physical safeguards",
    "reasonable and appropriate",
    "covered entity",
    "business associate",
    "protected health information",
    "phi",
    "individually identifiable health information",
}

# Паттерны для вопросов о принципах/концепциях (порядок важен - сначала более специфичные)
PRINCIPLE_PATTERNS = [
    r'what does\s+["\']([^"\']+)["\']\s+mean',  # Кавычки: "minimum necessary"
    r'what does\s+([A-Za-z\s]+?)\s+mean',       # Без кавычек: minimum necessary (захватывает до "mean")
    r'define\s+([A-Za-z\s]+?)(?:\s+(?:in|for|under|by|with|to|of|and|or|the|a|an|terminology|hipaa|context|regulations?|provisions?|mean)|$|\.|\?|!)', # define X
    r'what is\s+([A-Za-z\s]+?)(?:\s+(?:in|for|under|by|with|to|of|and|or|the|a|an|terminology|hipaa|context|regulations?|provisions?|mean)|$|\.|\?|!)', # what is X
    r'explain\s+([A-Za-z\s]+?)(?:\s+(?:in|for|under|by|with|to|of|and|or|the|a|an|terminology|hipaa|context|regulations?|provisions?|mean)|$|\.|\?|!)', # explain X
]


def extract_term_from_question(question: str) -> Optional[str]:
    """
    Извлекает термин из вопроса, используя паттерны.
    
    Args:
        question: Вопрос пользователя
    
    Returns:
        Извлеченный термин (в нижнем регистре) или None
    """
    question_lower = question.lower()
    
    for pattern in PRINCIPLE_PATTERNS:
        match = re.search(pattern, question_lower, re.IGNORECASE)
        if match:
            term = match.group(1).strip()
            # Убираем пунктуацию в конце
            term = re.sub(r'[.,;:!?]+$', '', term)
            # Убираем лишние слова в конце (in, for, under, etc.) - но только если они отделены пробелом
            term = re.sub(r'\s+(in|for|under|by|with|to|of|and|or|the|a|an|terminology|hipaa|context|regulations?|provisions?|mean)\s*$', '', term, flags=re.IGNORECASE)
            if term:
                term_normalized = term.lower().strip()
                # Проверяем, что извлеченный термин имеет смысл (хотя бы 3 символа)
                if len(term_normalized) >= 3:
                    return term_normalized
    
    return None


def is_regulatory_principle(term: str) -> bool:
    """
    Проверяет, является ли термин регуляторной концепцией.
    
    Args:
        term: Термин для проверки
    
    Returns:
        True, если термин в словаре регуляторных концепций
    """
    term_normalized = term.lower().strip()
    
    # Прямое совпадение
    if term_normalized in REGULATORY_CONCEPTS:
        return True
    
    # Частичное совпадение (например, "minimum necessary principle" содержит "minimum necessary")
    for concept in REGULATORY_CONCEPTS:
        if concept in term_normalized or term_normalized in concept:
            return True
    
    return False


def apply_classification_override(
    classification: QuestionClassification,
    question: str
) -> QuestionClassification:
    """
    Применяет post-classification override правила.
    
    Правила:
    - Если вопрос матчится под паттерн "what does X mean" / "define X"
    - И X в словаре regulatory concepts
    - То категория становится "regulatory_principle" (или "procedural / best practices" если новая категория не введена)
    
    Args:
        classification: Результат классификации от LLM
        question: Оригинальный вопрос пользователя
    
    Returns:
        Обновленная классификация (или исходная, если override не сработал)
    """
    # Извлекаем термин из вопроса
    term = extract_term_from_question(question)
    
    if not term:
        return classification
    
    # Проверяем, является ли термин регуляторной концепцией
    if not is_regulatory_principle(term):
        return classification
    
    # Если категория уже была "definition", переопределяем
    if classification.category == "definition":
        logger.info(
            f"Classification override: '{classification.category}' -> 'regulatory_principle' "
            f"(term: '{term}' is a regulatory concept)"
        )
        
        # Создаем новую классификацию с переопределенной категорией
        return QuestionClassification(
            category="regulatory_principle",
            confidence=min(classification.confidence + 0.1, 1.0),  # Немного повышаем уверенность
            reasoning=f"Override: '{term}' is a regulatory principle/concept, not a simple definition. "
                     f"Original: {classification.reasoning}"
        )
    
    return classification
