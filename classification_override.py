"""Post-classification override для применения доменных правил.

После классификации вопросов LLM применяются rule-based правила для переопределения категорий
на основе паттернов вопросов и словарей терминов.

Шаг 3: Добавлены router signals для определения require_citations, citation_mode, anchor_hint, scope_hint.
"""

import re
import logging
from typing import Optional, Tuple

from classifier import QuestionClassification

logger = logging.getLogger(__name__)


# Словарь регуляторных концепций/принципов (не простые определения)
# Примечание: "business associate", "covered entity" и другие формально определенные термины
# НЕ должны быть здесь - они являются определениями (definitions), а не принципами
REGULATORY_CONCEPTS = {
    "minimum necessary",
    "reasonable safeguards",
    "addressable implementation specification",
    "administrative safeguards",
    "technical safeguards",
    "physical safeguards",
    "reasonable and appropriate",
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

# Шаг 3.1: Regex для явного якоря в вопросе
EXPLICIT_ANCHOR_RE = re.compile(r"§\s*\d+\.\d+(?:\([a-z0-9ivx]+\))*", re.IGNORECASE)

# Шаг 3.2: Словарь слов, указывающих на необходимость цитирования
CITE_WORDS = [
    "cite",
    "citation",
    "quote",
    "exact text",
    "verbatim",
    "show the text",
    "show the regulation",
    "what does the regulation say",
    "what is the exact wording",
]

# Шаг 3.3: Строгие темы для определения anchor_hint и scope_hint
STRICT_TOPICS = {
    "law enforcement": {
        "anchor_hint": "§164.512(f)",
        "scope_hint": "law enforcement",
        "keywords": ["law enforcement", "police", "court", "warrant", "subpoena"]
    },
    "suspect_fugitive": {
        "anchor_hint": "§164.512(f)(2)",
        "scope_hint": "suspect/fugitive/witness/missing person",
        "keywords": ["suspect", "fugitive", "material witness", "missing person", "identify or locate", "identify", "locate"]
    },
    "victim": {
        "anchor_hint": "§164.512(f)(3)",
        "scope_hint": "crime victim",
        "keywords": ["victim"],
        "required_context": ["crime", "suspected victim"]  # Должен быть вместе с victim
    }
}


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


def _detect_router_signals(question: str) -> Tuple[Optional[str], bool, Optional[str], Optional[str]]:
    """
    Шаг 3: Определяет router signals из вопроса.
    
    Args:
        question: Текст вопроса
    
    Returns:
        Tuple из (explicit_anchor, has_cite_words, anchor_hint, scope_hint)
    """
    question_lower = question.lower()
    
    # Шаг 3.1: Проверяем явный anchor в вопросе
    explicit_anchor = None
    anchor_match = EXPLICIT_ANCHOR_RE.search(question)
    if anchor_match:
        explicit_anchor = anchor_match.group(0)
        # Нормализуем: убираем пробелы после §
        explicit_anchor = re.sub(r'§\s+', '§', explicit_anchor)
        # Убираем все пробелы между символами
        explicit_anchor = re.sub(r'\s+', '', explicit_anchor)
        logger.info(f"Router signal: explicit_anchor='{explicit_anchor}'")
    
    # Шаг 3.2: Проверяем cite words
    has_cite_words = any(cite_word in question_lower for cite_word in CITE_WORDS)
    if has_cite_words:
        logger.info(f"Router signal: has_cite_words=True")
    
    # Шаг 3.3: Определяем strict topics для anchor_hint и scope_hint
    anchor_hint = None
    scope_hint = None
    
    # Проверяем victim + crime/suspected victim -> §164.512(f)(3)
    if "victim" in question_lower:
        if "crime" in question_lower or "suspected victim" in question_lower:
            anchor_hint = STRICT_TOPICS["victim"]["anchor_hint"]
            scope_hint = STRICT_TOPICS["victim"]["scope_hint"]
            logger.info(f"Router signal: strict_topic='victim', anchor_hint='{anchor_hint}', scope_hint='{scope_hint}'")
    
    # Проверяем suspect/fugitive/witness/missing person -> §164.512(f)(2)
    if not anchor_hint:  # Только если еще не определили
        for keyword in STRICT_TOPICS["suspect_fugitive"]["keywords"]:
            if keyword in question_lower:
                anchor_hint = STRICT_TOPICS["suspect_fugitive"]["anchor_hint"]
                scope_hint = STRICT_TOPICS["suspect_fugitive"]["scope_hint"]
                logger.info(f"Router signal: strict_topic='suspect_fugitive', anchor_hint='{anchor_hint}', scope_hint='{scope_hint}'")
                break
    
    # Проверяем law enforcement -> §164.512(f)
    if not anchor_hint:  # Только если еще не определили
        for keyword in STRICT_TOPICS["law enforcement"]["keywords"]:
            if keyword in question_lower:
                anchor_hint = STRICT_TOPICS["law enforcement"]["anchor_hint"]
                scope_hint = STRICT_TOPICS["law enforcement"]["scope_hint"]
                logger.info(f"Router signal: strict_topic='law enforcement', anchor_hint='{anchor_hint}', scope_hint='{scope_hint}'")
                break
    
    # Сохраняем исходный scope_hint до переопределения explicit_anchor
    strict_topic_scope_hint = scope_hint
    
    # Если нашли explicit_anchor, используем его как anchor_hint (приоритет)
    if explicit_anchor:
        anchor_hint = explicit_anchor
        scope_hint = "explicit_anchor"  # Маркер для Rule 4.1
        logger.info(f"Router signal: using explicit_anchor as anchor_hint='{anchor_hint}'")
    
    return explicit_anchor, has_cite_words, anchor_hint, scope_hint, strict_topic_scope_hint


def apply_classification_override(
    classification: QuestionClassification,
    question: str
) -> tuple[QuestionClassification, Optional[str]]:
    """
    Применяет post-classification override правила.
    
    Правила:
    - Если вопрос матчится под паттерн "what does X mean" / "define X"
    - И X в словаре regulatory concepts
    - То категория становится "regulatory_principle"
    
    Шаг 3: Также определяет router signals (require_citations, citation_mode, anchor_hint, scope_hint).
    Шаг 4: Реализует правила переопределения citation intent с приоритетами.
    
    Args:
        classification: Результат классификации от LLM
        question: Оригинальный вопрос пользователя
    
    Returns:
        Tuple из (обновленная классификация, concept_term)
        - Если override не сработал: (исходная классификация, None)
        - Если override сработал: (новая классификация, извлеченный термин)
    """
    # Шаг 3: Определяем router signals
    explicit_anchor, has_cite_words, anchor_hint, scope_hint, strict_topic_scope_hint = _detect_router_signals(question)
    
    # Шаг 5: Сначала применяем regulatory_principle override (term extraction)
    # Извлекаем термин из вопроса для regulatory_principle override
    term = extract_term_from_question(question)
    concept_term = None
    
    if term and is_regulatory_principle(term):
        # Если категория уже была "definition", переопределяем на "regulatory_principle"
        if classification.category == "definition":
            original_category = classification.category
            logger.info(
                f"Regulatory principle override: original_category='{original_category}' -> overridden_category='regulatory_principle' "
                f"(term: '{term}' is a regulatory concept)"
            )
            
            # Создаем новую классификацию с переопределенной категорией
            # Пока не применяем citation intent rules - они будут применены после
            classification = QuestionClassification(
                category="regulatory_principle",
                confidence=min(classification.confidence + 0.1, 1.0),  # Немного повышаем уверенность
                reasoning=f"Override: '{term}' is a regulatory principle/concept, not a simple definition. "
                         f"Original: {classification.reasoning}",
                require_citations=classification.require_citations,
                citation_mode=classification.citation_mode,
                anchor_hint=classification.anchor_hint,
                scope_hint=classification.scope_hint
            )
            concept_term = term
    
    # Шаг 4: Затем применяем правила переопределения citation intent (в порядке приоритетов)
    updated_fields = {}
    category_changed = False
    original_category = classification.category
    is_regulatory_principle_category = (classification.category == "regulatory_principle")
    
    # Rule 4.1 - Explicit anchor wins (strict) - самый высокий приоритет
    # explicit_anchor всегда wins, даже для regulatory_principle
    if explicit_anchor:
        logger.info(f"Rule 4.1: Explicit anchor found -> strict citation mode, category='citation-required'")
        updated_fields["require_citations"] = True
        updated_fields["citation_mode"] = "strict"
        updated_fields["anchor_hint"] = explicit_anchor
        updated_fields["scope_hint"] = "explicit_anchor"
        if classification.category != "citation-required":
            updated_fields["category"] = "citation-required"
            category_changed = True
            logger.info(f"Rule 4.1: Category override: '{original_category}' -> 'citation-required'")
    
    # Rule 4.2 - Cite words + strict topic -> strict + anchor_hint
    # Для regulatory_principle это тоже форсит citation-required
    elif has_cite_words and strict_topic_scope_hint and strict_topic_scope_hint != "explicit_anchor":
        # strict_topic_scope_hint не None означает, что был найден strict topic
        logger.info(f"Rule 4.2: Cite words + strict topic -> strict citation mode, category='citation-required'")
        updated_fields["require_citations"] = True
        updated_fields["citation_mode"] = "strict"
        updated_fields["anchor_hint"] = anchor_hint  # Уже установлен из strict topic
        updated_fields["scope_hint"] = strict_topic_scope_hint
        if classification.category != "citation-required":
            updated_fields["category"] = "citation-required"
            category_changed = True
            logger.info(f"Rule 4.2: Category override: '{original_category}' -> 'citation-required'")
    
    # Rule 4.3 - Cite words only -> quoted, category not changed
    # Для regulatory_principle: citation_mode="quoted", но category остается "regulatory_principle"
    elif has_cite_words:
        if is_regulatory_principle_category:
            logger.info(f"Rule 4.3: Cite words only -> quoted citation mode for regulatory_principle, category unchanged")
            updated_fields["require_citations"] = True
            updated_fields["citation_mode"] = "quoted"
            # anchor_hint и scope_hint оставляем как есть (могут быть установлены из router signals)
            # category НЕ меняем (остается regulatory_principle)
        else:
            logger.info(f"Rule 4.3: Cite words only -> quoted citation mode, category unchanged")
            updated_fields["require_citations"] = True
            updated_fields["citation_mode"] = "quoted"
            updated_fields["anchor_hint"] = None
            updated_fields["scope_hint"] = "cite_requested"
            # category НЕ меняем (важно!)
    
    # Если есть изменения в router signals, обновляем classification
    if updated_fields:
        logger.info(f"Citation intent override: {updated_fields}")
        classification_dict = classification.model_dump()
        classification_dict.update(updated_fields)
        classification = QuestionClassification(**classification_dict)
    
    # Возвращаем результат с concept_term (если был найден)
    return classification, concept_term
