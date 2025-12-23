"""Ретривер для вопросов типа 'citation-required' - строгое цитирование."""

from typing import List, Optional, Dict, Any
import logging
import re

from .base import BaseRetriever, CandidateChunk
from .utils import (
    get_db_connection, normalize_fts_scores, merge_candidates,
    calculate_final_score, build_fts_query
)
from embeddings import get_embedding_client

logger = logging.getLogger(__name__)


# Шаг 1: Классификация "colon type" для обработки incomplete цитат
def ends_with_colon(text: str) -> bool:
    """
    Проверяет, заканчивается ли текст на двоеточие.
    
    Args:
        text: Текст для проверки
    
    Returns:
        True, если текст заканчивается на ':'
    """
    if not text:
        return False
    return text.rstrip().endswith(":")


def is_list_introducer(text: str) -> bool:
    """
    Проверяет, является ли текст вводом к списку (list introducer).
    
    Возвращает True если текст явно вводит список и должен продолжаться:
    - содержит (case-insensitive) любые из паттернов:
      * "may disclose protected health information:"
      * "the following:"
      * "as follows:"
      * "includes:"
      * "pursuant to the following:"
    
    Важно: "(x) Standard:" НЕ является list introducer (это заголовок стандарта).
    
    Args:
        text: Текст для проверки
    
    Returns:
        True, если текст является list introducer
    """
    if not text:
        return False
    
    # Шаг 1: Проверяем, не является ли это стандартным заголовком
    # "(x) Standard:" никогда не считается list introducer
    if re.search(r"\([a-z]\)\s+Standard:", text, re.IGNORECASE):
        return False
    
    text_lower = text.lower()
    
    list_introducer_patterns = [
        r"may disclose protected health information:",
        r"the following:",
        r"as follows:",
        r"includes:",
        r"pursuant to the following:",
    ]
    
    for pattern in list_introducer_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def is_standard_header(text: str) -> bool:
    """
    Проверяет, является ли текст заголовочным "Standard:" и его допустимо цитировать целиком.
    
    True если:
    - содержит паттерн "(x) Standard:" (regex r"\([a-z]\)\s+Standard:", case-insensitive)
    - и НЕ является list_introducer
    
    Args:
        text: Текст для проверки
    
    Returns:
        True, если текст является standard header
    """
    if not text:
        return False
    
    # Проверяем паттерн "(x) Standard:"
    standard_pattern = r"\([a-z]\)\s+Standard:"
    if not re.search(standard_pattern, text, re.IGNORECASE):
        return False
    
    # НЕ должен быть list_introducer
    if is_list_introducer(text):
        return False
    
    return True


# Adaptive widening helpers для CitationRetriever
def strip_last_paren(anchor_prefix: str) -> Optional[str]:
    """
    Удаляет последнюю скобку с содержимым из anchor prefix.
    
    Примеры:
        "§164.512(f)(1)" -> "§164.512(f)"
        "§164.512(f)" -> "§164.512"
        "§164.512" -> None (нельзя расширить дальше)
    
    Args:
        anchor_prefix: Anchor prefix (например, "§164.512(f)(1)")
    
    Returns:
        Расширенный prefix или None, если нельзя расширить
    """
    if not anchor_prefix:
        return None
    
    # Ищем последнюю скобку с содержимым: (x) или (x)(y)
    # Паттерн: \(([a-z0-9]+)\) в конце строки
    match = re.search(r'\([a-z0-9]+\)$', anchor_prefix)
    if match:
        # Удаляем последнюю скобку
        return anchor_prefix[:match.start()]
    
    return None


def strip_to_section(anchor_prefix: str) -> Optional[str]:
    """
    Расширяет anchor prefix до уровня секции (удаляет все подпункты).
    
    Примеры:
        "§164.512(f)(1)" -> "§164.512"
        "§164.512(f)" -> "§164.512"
        "§164.512" -> "§164.512" (без изменений)
    
    Args:
        anchor_prefix: Anchor prefix (например, "§164.512(f)(1)")
    
    Returns:
        Prefix на уровне секции или None
    """
    if not anchor_prefix:
        return None
    
    # Ищем паттерн §XXX.XXX (секция)
    match = re.search(r'^§\d+\.\d+', anchor_prefix)
    if match:
        return match.group(0)
    
    return None


class CitationRetriever(BaseRetriever):
    """
    Ретривер для вопросов типа 'citation-required' - строгое цитирование.
    
    Для вопросов, где явно требуется цитирование, возвращает атомарные параграфы
    в виде anchor + text_raw (дословная цитата) без пересказа, интерпретации, выводов.
    """
    
    # Маппинг тем на anchor prefix
    TOPIC_ANCHOR_PREFIXES = {
        "law enforcement": "§164.512",
        "family": "§164.510",
        "public health": "§164.512",
        "disclosure": "§164.502",
    }
    
    def __init__(self, db_connection=None):
        """
        Инициализация ретривера цитирования.
        
        Args:
            db_connection: Подключение к PostgreSQL (если None, создается новое)
        """
        self.db = db_connection or get_db_connection()
        self.embedding_client = get_embedding_client()
    
    async def retrieve(
        self,
        question_embedding: List[float],
        max_results: int = 6,
        question: Optional[str] = None,
        doc_id: Optional[str] = None,
        anchor_prefix: Optional[str] = None,
        k: int = 6,
        seed_k: int = 6,
        expand_section: bool = True,
        min_relevance: Optional[float] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Поиск для строгого цитирования.
        
        Args:
            question_embedding: Эмбеддинг вопроса (размерность 4096)
            max_results: Максимальное количество результатов (по умолчанию 6)
            question: Текст вопроса
            doc_id: ID документа (по умолчанию 'hipaa-reg-2013-03-26')
            anchor_prefix: Префикс anchor для фильтрации (например, "§164.512")
            k: Количество итоговых цитат (по умолчанию 6)
            seed_k: Количество seed-чанков (по умолчанию 6)
            expand_section: Расширить внутри секции (по умолчанию True)
            min_relevance: Минимальный порог final_score (опционально)
        
        Returns:
            Список словарей с информацией о цитатах (anchor + text_raw)
        """
        doc_id = doc_id or "hipaa-reg-2013-03-26"
        question = question or ""
        k = min(k, max_results, 10)  # Максимум 10 цитат
        
        # Step 0: Зафиксировать "юрисдикцию" (scope) - anchor_prefix как "prefer", не жесткий
        if not anchor_prefix:
            anchor_prefix = self.infer_anchor_prefix(question)
        
        # Adaptive widening: если по строгому prefix ничего не найдено, расширяем scope
        effective_prefix = anchor_prefix
        original_prefix = anchor_prefix
        
        # Генерируем список префиксов для widening (от самого узкого к широкому)
        widening_prefixes = [anchor_prefix] if anchor_prefix else []
        if anchor_prefix:
            # Пробуем расширить до родительского уровня
            parent_prefix = strip_last_paren(anchor_prefix)
            if parent_prefix:
                widening_prefixes.append(parent_prefix)
                # Пробуем расширить до уровня секции
                section_prefix = strip_to_section(anchor_prefix)
                if section_prefix and section_prefix != parent_prefix:
                    widening_prefixes.append(section_prefix)
        
        logger.info(f"CitationRetriever: question='{question[:50]}...', original_anchor_prefix={original_prefix}, widening_prefixes={widening_prefixes}, k={k}")
        
        # Шаг 2: Hard fallback to broader prefix - пробуем каждый prefix, пока не получим валидные результаты
        # Это цикл по widening_prefixes, который продолжается до получения непустых results
        final_results = []
        tried_prefixes = []
        
        for attempt_prefix in widening_prefixes:
            tried_prefixes.append(attempt_prefix)
            test_anchor_like = f"{attempt_prefix}%"
            
            logger.info(f"Attempting retrieval with prefix: {attempt_prefix}")
            
            # Step 1: Vector search
            test_vector = await self._vector_search_citation(
                question_embedding=question_embedding,
                doc_id=doc_id,
                anchor_like=test_anchor_like,
                limit=50
            )
            
            # Step 2: FTS search
            test_fts = []
            if question and question.strip():
                boost_words = [
                    "law enforcement", "police", "court", "warrant",
                    "subpoena", "administrative request", "disclosure", "disclose"
                ]
                fts_query = build_fts_query(question, boost_words)
                test_fts = await self._fts_search_citation(
                    fts_query=fts_query,
                    doc_id=doc_id,
                    anchor_like=test_anchor_like,
                    limit=50
                )
            
            total_found = len(test_vector) + len(test_fts)
            
            if total_found == 0:
                logger.debug(f"No candidates found for prefix {attempt_prefix}, trying next level...")
                continue
            
            # Нашли кандидатов - обрабатываем их
            effective_prefix = attempt_prefix
            effective_anchor_like = test_anchor_like
            vector_candidates = test_vector
            fts_candidates = test_fts
            
            if attempt_prefix != original_prefix:
                logger.info(f"Widening scope: {original_prefix} -> {attempt_prefix} (found {total_found} candidates: {len(test_vector)} vector + {len(test_fts)} FTS)")
            else:
                logger.info(f"Found {total_found} candidates with original prefix {attempt_prefix} ({len(test_vector)} vector + {len(test_fts)} FTS)")
            
            # Обрабатываем кандидатов для этого prefix
            # Step 3: Merge + selection
            merged = self._merge_and_score_citation(vector_candidates, fts_candidates)
            
            # Фильтруем по min_relevance если задан
            if min_relevance:
                merged = [c for c in merged if c.final_score >= min_relevance]
            
            # Выбираем top seed_k
            seeds = merged[:seed_k]
            logger.info(f"Выбрано seed-чанков: {len(seeds)}")
            
            all_results = list(seeds)
            
            # Step 4: Coverage expansion (собрать подпункты)
            if expand_section and effective_anchor_like:
                expanded = await self._expand_coverage(seeds, doc_id, effective_anchor_like, k)
                logger.info(f"Найдено expanded-чанков: {len(expanded)}")
                all_results.extend(expanded)
            
            # Дедупликация и выбор top-k
            final_results = self._dedup_and_select_citation(all_results, k)
            
            logger.info(f"Финальных результатов (до фильтрации incomplete) для prefix {attempt_prefix}: {len(final_results)}")
            
            # Шаг 2: Классификация и фильтрация чанков с двоеточием
            list_introducer_seeds = []
            standard_header_results = []
            colon_unknown_type = []
            complete_results = []
            
            for candidate in final_results:
                if ends_with_colon(candidate.text_raw):
                    if is_standard_header(candidate.text_raw):
                        standard_header_results.append(candidate)
                        logger.info(f"Standard header detected: anchor={candidate.anchor}, text_raw='{candidate.text_raw[:50]}...'")
                    elif is_list_introducer(candidate.text_raw):
                        list_introducer_seeds.append(candidate)
                        logger.info(f"List introducer detected: anchor={candidate.anchor}, text_raw='{candidate.text_raw[:50]}...'")
                    else:
                        colon_unknown_type.append(candidate)
                        logger.warning(f"Colon unknown type: anchor={candidate.anchor}, text_raw='{candidate.text_raw[:50]}...'")
                else:
                    complete_results.append(candidate)
            
            # Шаг 3 и 4: Coverage completion для list introducer seeds
            expanded_from_list_introducers = []
            incomplete_warnings = []
            
            for list_introducer_seed in list_introducer_seeds:
                if not list_introducer_seed.anchor:
                    incomplete_warnings.append(f"list_introducer_no_anchor: {list_introducer_seed.chunk_id}")
                    continue
                
                children = await self._expand_incomplete_seed(
                    seed_anchor=list_introducer_seed.anchor,
                    doc_id=doc_id,
                    limit=12
                )
                
                if children:
                    list_introducer_seed.explanation = "list_introducer_header"
                    expanded_from_list_introducers.append(list_introducer_seed)
                    for child in children[:6]:
                        child.explanation = f"expanded_from_list_introducer:{list_introducer_seed.anchor}"
                        expanded_from_list_introducers.append(child)
                    logger.info(f"Expanded list introducer {list_introducer_seed.anchor}: added seed + {min(len(children), 6)} child subparagraphs")
                else:
                    if is_standard_header(list_introducer_seed.text_raw):
                        standard_header_results.append(list_introducer_seed)
                        logger.info(f"Fallback A: list introducer {list_introducer_seed.anchor} is also standard header - returning as-is")
                    else:
                        siblings = await self._get_sibling_chunks(
                            seed=list_introducer_seed,
                            doc_id=doc_id,
                            anchor_like=effective_anchor_like,
                            limit=12
                        )
                        if siblings:
                            for sibling in siblings:
                                sibling.explanation = f"expanded_from_list_introducer_siblings:{list_introducer_seed.anchor}"
                                expanded_from_list_introducers.append(sibling)
                            logger.info(f"Fallback B: list introducer {list_introducer_seed.anchor} - found {len(siblings)} sibling chunks")
                        else:
                            # Fallback C: нет детей и siblings - это нормально, будет расширение до более широкого prefix
                            logger.warning(f"List introducer {list_introducer_seed.anchor} has no children and no siblings - will try broader prefix")
            
            # Объединяем все результаты
            all_final_results = (
                list(complete_results) + 
                standard_header_results + 
                expanded_from_list_introducers + 
                colon_unknown_type
            )
            
            # Дедупликация
            seen_chunk_ids = set()
            deduped_results = []
            for candidate in all_final_results:
                if candidate.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(candidate.chunk_id)
                    deduped_results.append(candidate)
            
            deduped_results.sort(key=lambda x: x.anchor or "")
            final_results = deduped_results
            
            logger.info(f"Финальных результатов (после фильтрации incomplete и expansion) для prefix {attempt_prefix}: {len(final_results)}")
            
            # Проверяем, если остался только standard_header без дочерних подпунктов - расширяемся до §164.512
            if final_results and len(final_results) == 1:
                single_result = final_results[0]
                # Проверяем, является ли это standard_header (паттерн "(x) Standard:")
                if single_result.anchor and single_result.text_raw:
                    is_standard_header_only = is_standard_header(single_result.text_raw)
                    # Проверяем, что anchor не равен §164.512 (чтобы не зациклиться)
                    # И что это не самый широкий prefix (чтобы не расширяться бесконечно)
                    if is_standard_header_only and single_result.anchor != "§164.512" and attempt_prefix != "§164.512":
                        logger.info(f"Only standard header found ({single_result.anchor}), expanding to broader prefix (current: {attempt_prefix})")
                        # Продолжаем цикл, чтобы попробовать более широкий prefix
                        continue
            
            # Если получили валидные результаты - останавливаемся
            if final_results:
                logger.info(f"Successfully retrieved {len(final_results)} results with prefix {attempt_prefix}")
                break
            else:
                logger.warning(f"No valid results for prefix {attempt_prefix} after processing, trying next level...")
        
        # Шаг 3: Если все еще пусто - fallback на §164.512 целиком
        if not final_results:
            logger.warning(f"No results after trying all prefixes: {tried_prefixes}, attempting fallback to §164.512")
            final_results = await self._fallback_to_section_164_512(doc_id, question_embedding, question, k)
        
        # Конвертируем в формат API
        results = []
        for candidate in final_results:
            result_dict = {
                "chunk_id": candidate.chunk_id,
                "anchor": candidate.anchor,
                "text_raw": candidate.text_raw,
                "section_id": candidate.section_id,
                "section_number": candidate.section_number,
                "section_title": candidate.section_title,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "scores": {
                    "vector_score": candidate.vector_score,
                    "fts_score": candidate.fts_score,
                    "final_score": candidate.final_score
                },
                "explanation": getattr(candidate, 'explanation', 'citation retrieval')
            }
            
            explanation = getattr(candidate, 'explanation', '')
            if explanation:
                if explanation.startswith('expanded_from_list_introducer:'):
                    expanded_from_anchor = explanation.split(':', 1)[1]
                    result_dict["flags"] = {"expanded_from": expanded_from_anchor, "type": "list_introducer_child"}
                elif explanation.startswith('expanded_from_list_introducer_siblings:'):
                    expanded_from_anchor = explanation.split(':', 1)[1]
                    result_dict["flags"] = {"expanded_from": expanded_from_anchor, "type": "sibling"}
                elif explanation == "list_introducer_header":
                    result_dict["flags"] = {"type": "list_introducer_header"}
            
            results.append(result_dict)
        
        logger.info(f"Final results count: {len(results)}")
        return results
        
        # Step 3: Merge + selection
        merged = self._merge_and_score_citation(vector_candidates, fts_candidates)
        
        # Фильтруем по min_relevance если задан
        if min_relevance:
            merged = [c for c in merged if c.final_score >= min_relevance]
        
        # Выбираем top seed_k
        seeds = merged[:seed_k]
        logger.info(f"Выбрано seed-чанков: {len(seeds)}")
        
        all_results = list(seeds)
        
        # Step 4: Coverage expansion (собрать подпункты)
        if expand_section and effective_anchor_like:
            expanded = await self._expand_coverage(seeds, doc_id, effective_anchor_like, k)
            logger.info(f"Найдено expanded-чанков: {len(expanded)}")
            all_results.extend(expanded)
        
        # Дедупликация и выбор top-k
        final_results = self._dedup_and_select_citation(all_results, k)
        
        logger.info(f"Финальных результатов (до фильтрации incomplete): {len(final_results)}")
        
        # Шаг 2: Классификация и фильтрация чанков с двоеточием
        list_introducer_seeds = []  # Нуждаются в expansion
        standard_header_results = []  # Допустимые цитаты (Standard:)
        colon_unknown_type = []  # Неизвестный тип, но оставляем (fail-soft)
        complete_results = []
        
        for candidate in final_results:
            if ends_with_colon(candidate.text_raw):
                # Классифицируем тип двоеточия
                if is_standard_header(candidate.text_raw):
                    # Standard header - допустимая цитата, оставляем
                    standard_header_results.append(candidate)
                    logger.info(f"Standard header detected: anchor={candidate.anchor}, text_raw='{candidate.text_raw[:50]}...'")
                elif is_list_introducer(candidate.text_raw):
                    # List introducer - нуждается в expansion
                    list_introducer_seeds.append(candidate)
                    logger.info(f"List introducer detected: anchor={candidate.anchor}, text_raw='{candidate.text_raw[:50]}...'")
                else:
                    # Неизвестный тип - оставляем (fail-soft), но логируем
                    colon_unknown_type.append(candidate)
                    logger.warning(f"Colon unknown type: anchor={candidate.anchor}, text_raw='{candidate.text_raw[:50]}...'")
            else:
                # Полные цитаты добавляем в финальный output
                complete_results.append(candidate)
        
        if list_introducer_seeds:
            logger.info(f"Found {len(list_introducer_seeds)} list introducers (needs expansion), kept {len(complete_results)} complete quotes")
        if standard_header_results:
            logger.info(f"Found {len(standard_header_results)} standard headers (allowed as-is)")
        if colon_unknown_type:
            logger.info(f"Found {len(colon_unknown_type)} colon unknown type (kept as fail-soft)")
        
        # Шаг 3 и 4: Coverage completion для list introducer seeds
        expanded_from_list_introducers = []
        incomplete_warnings = []
        
        for list_introducer_seed in list_introducer_seeds:
            if not list_introducer_seed.anchor:
                incomplete_warnings.append(f"list_introducer_no_anchor: {list_introducer_seed.chunk_id}")
                continue
            
            # Ищем дочерние подпункты для list introducer seed
            children = await self._expand_incomplete_seed(
                seed_anchor=list_introducer_seed.anchor,
                doc_id=doc_id,
                limit=12
            )
            
            if children:
                # Добавляем seed + дочерние подпункты (seed + 2-6 детей)
                # Добавляем seed (заголовок списка)
                list_introducer_seed.explanation = "list_introducer_header"
                expanded_from_list_introducers.append(list_introducer_seed)
                
                # Добавляем дочерние подпункты (2-6 лучших)
                for child in children[:6]:
                    child.explanation = f"expanded_from_list_introducer:{list_introducer_seed.anchor}"
                    expanded_from_list_introducers.append(child)
                
                logger.info(f"Expanded list introducer {list_introducer_seed.anchor}: added seed + {min(len(children), 6)} child subparagraphs")
            else:
                # Шаг 4: Fail-soft fallback - если детей нет
                # Fallback A: если это standard header - вернуть seed целиком
                # (Примечание: это маловероятно, так как is_standard_header уже проверяет !is_list_introducer,
                # но оставляем для безопасности)
                if is_standard_header(list_introducer_seed.text_raw):
                    standard_header_results.append(list_introducer_seed)
                    logger.info(f"Fallback A: list introducer {list_introducer_seed.anchor} is also standard header - returning as-is")
                else:
                    # Fallback B: вернуть sibling chunks из той же секции
                    siblings = await self._get_sibling_chunks(
                        seed=list_introducer_seed,
                        doc_id=doc_id,
                        anchor_like=effective_anchor_like,
                        limit=12
                    )
                    
                    if siblings:
                        for sibling in siblings:
                            sibling.explanation = f"expanded_from_list_introducer_siblings:{list_introducer_seed.anchor}"
                            expanded_from_list_introducers.append(sibling)
                        logger.info(f"Fallback B: list introducer {list_introducer_seed.anchor} - found {len(siblings)} sibling chunks")
                    else:
                        # Fallback C: нет детей и siblings - это нормально, будет расширение до более широкого prefix
                        logger.warning(f"List introducer {list_introducer_seed.anchor} has no children and no siblings - will try broader prefix")
        
        # Объединяем все результаты: complete + standard headers + expanded from list introducers + colon unknown type
        all_final_results = (
            list(complete_results) + 
            standard_header_results + 
            expanded_from_list_introducers + 
            colon_unknown_type
        )
        
        # Дедупликация по chunk_id (на случай если дочерний подпункт уже был в complete_results)
        seen_chunk_ids = set()
        deduped_results = []
        for candidate in all_final_results:
            if candidate.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(candidate.chunk_id)
                deduped_results.append(candidate)
        
        # Сортируем по anchor (как раньше)
        deduped_results.sort(key=lambda x: x.anchor or "")
        
        # Используем deduped_results для финального output
        final_results = deduped_results
        
        logger.info(f"Финальных результатов (после фильтрации incomplete и expansion): {len(final_results)}")
        if incomplete_warnings:
            logger.warning(f"Incomplete seed warnings: {incomplete_warnings}")
        
        # Конвертируем в формат API (строго anchor + text_raw)
        results = []
        for candidate in final_results:
            result_dict = {
                "chunk_id": candidate.chunk_id,
                "anchor": candidate.anchor,
                "text_raw": candidate.text_raw,
                # Метаданные (опционально, не для user-facing ответа)
                "section_id": candidate.section_id,
                "section_number": candidate.section_number,
                "section_title": candidate.section_title,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "scores": {
                    "vector_score": candidate.vector_score,
                    "fts_score": candidate.fts_score,
                    "final_score": candidate.final_score
                },
                "explanation": getattr(candidate, 'explanation', 'citation retrieval')
            }
            
            # Добавляем флаги для expanded chunks
            explanation = getattr(candidate, 'explanation', '')
            if explanation:
                if explanation.startswith('expanded_from_list_introducer:'):
                    expanded_from_anchor = explanation.split(':', 1)[1]
                    result_dict["flags"] = {"expanded_from": expanded_from_anchor, "type": "list_introducer_child"}
                elif explanation.startswith('expanded_from_list_introducer_siblings:'):
                    expanded_from_anchor = explanation.split(':', 1)[1]
                    result_dict["flags"] = {"expanded_from": expanded_from_anchor, "type": "sibling"}
                elif explanation == "list_introducer_header":
                    result_dict["flags"] = {"type": "list_introducer_header"}
            
            results.append(result_dict)
        
        # Warnings логируются, но не возвращаются как error message
        # Если results пустой - fallback на §164.512 уже обработан выше
        if incomplete_warnings:
            logger.info(f"Incomplete seed warnings (handled by fallback): {incomplete_warnings}")
        
        return results
    
    async def _fallback_to_section_164_512(
        self,
        doc_id: str,
        question_embedding: List[float],
        question: str,
        k: int
    ) -> List[CandidateChunk]:
        """
        Fallback: возвращает §164.512 целиком, если нет подпунктов.
        
        Сценарий A: В базе есть chunk с anchor §164.512 (section-level или atomic)
        Сценарий B: В базе нет §164.512, но есть много §164.512(a), §164.512(b) и т.д.
        
        Args:
            doc_id: ID документа
            question_embedding: Эмбеддинг вопроса
            question: Текст вопроса
            k: Количество результатов
        
        Returns:
            Список CandidateChunk с результатами по §164.512
        """
        logger.info("Attempting fallback to §164.512")
        
        with self.db.cursor() as cur:
            # Сценарий A: Ищем section-level или atomic chunk с точным anchor §164.512
            cur.execute("""
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.text_raw,
                    c.page_start,
                    c.page_end,
                    CASE 
                        WHEN c.embedding IS NOT NULL THEN 1 - (c.embedding <=> %s::vector)
                        ELSE 0.5
                    END AS vector_score,
                    0.0 AS fts_score,
                    CASE 
                        WHEN c.embedding IS NOT NULL THEN 1 - (c.embedding <=> %s::vector)
                        ELSE 0.5
                    END AS final_score
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.anchor = '§164.512'
                ORDER BY c.granularity = 'section' DESC, c.embedding <=> %s::vector NULLS LAST
                LIMIT 1
            """, (question_embedding, question_embedding, doc_id, question_embedding))
            
            row = cur.fetchone()
            if row:
                logger.info("Found section-level or atomic chunk with anchor §164.512")
                return [CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[5],
                    page_start=row[6],
                    page_end=row[7],
                    vector_score=row[8] or 0.5,
                    fts_score=0.0,
                    final_score=row[10] or 0.5
                )]
            
            # Сценарий B: Агрегат из atomic chunks anchor LIKE '§164.512%'
            logger.info("No exact §164.512 chunk found, retrieving aggregate from §164.512%")
            cur.execute("""
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.text_raw,
                    c.page_start,
                    c.page_end,
                    CASE 
                        WHEN c.embedding IS NOT NULL THEN 1 - (c.embedding <=> %s::vector)
                        ELSE 0.5
                    END AS vector_score,
                    0.0 AS fts_score,
                    CASE 
                        WHEN c.embedding IS NOT NULL THEN 1 - (c.embedding <=> %s::vector)
                        ELSE 0.5
                    END AS final_score
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.anchor LIKE '§164.512%'
                ORDER BY c.anchor, c.embedding <=> %s::vector NULLS LAST
                LIMIT %s
            """, (question_embedding, question_embedding, doc_id, question_embedding, min(k * 2, 25)))
            
            rows = cur.fetchall()
            if rows:
                logger.info(f"Found {len(rows)} atomic chunks with anchor LIKE '§164.512%'")
                candidates = []
                for row in rows:
                    candidates.append(CandidateChunk(
                        chunk_id=row[0],
                        anchor=row[1],
                        section_id=row[2],
                        section_number=row[3],
                        section_title=row[4],
                        text_raw=row[5],
                        page_start=row[6],
                        page_end=row[7],
                        vector_score=row[8] or 0.5,
                        fts_score=0.0,
                        final_score=row[10] or 0.5
                    ))
                return candidates[:k]
            
            logger.warning("No chunks found for fallback to §164.512")
            return []
    
    def infer_anchor_prefix(self, question: str) -> str:
        """
        Определяет точный anchor prefix по вопросу (scope parsing для подпунктов).
        
        Шаг 2.1: Более точное определение prefix для law enforcement:
        - Если встречается law enforcement -> §164.512(f)
        - Если встречается suspect/fugitive/material witness/missing person/identify or locate -> §164.512(f)(2)
        - Если встречается victim и (crime или suspected victim) -> §164.512(f)(3)
        - Если в вопросе явно есть строка вида §164.512(f) - уважать её
        - Иначе fallback: §164.512
        
        Args:
            question: Текст вопроса
        
        Returns:
            anchor prefix (например, "§164.512(f)" или "§164.512")
        """
        if not question:
            return "§164.512"  # Дефолт
        
        question_lower = question.lower()
        
        # Шаг 2.1.4: Проверяем явное указание anchor в вопросе (самый сильный сигнал)
        # Паттерн должен захватывать §164.512(f), §164.512(f)(2), §164.512(f)(3) и т.д.
        explicit_anchor_match = re.search(r'§\s*164\.512\s*(?:\([a-z0-9]+\)(?:\([0-9]+\))?)+', question, re.IGNORECASE)
        if explicit_anchor_match:
            anchor_found = explicit_anchor_match.group(0)
            # Нормализуем (убираем пробелы после § и между скобками)
            anchor_found = re.sub(r'§\s+', '§', anchor_found)
            anchor_found = re.sub(r'\s+', '', anchor_found)  # Убираем все пробелы
            logger.info(f"Found explicit anchor in question: {anchor_found}")
            return anchor_found
        
        # Шаг 2.1.3: Проверяем victim + crime/suspected victim -> §164.512(f)(3)
        if "victim" in question_lower:
            if "crime" in question_lower or "suspected victim" in question_lower:
                logger.info("Inferred anchor prefix: §164.512(f)(3) (victim + crime)")
                return "§164.512(f)(3)"
        
        # Шаг 2.1.2: Проверяем suspect/fugitive/material witness/missing person/identify or locate -> §164.512(f)(2)
        f2_keywords = [
            "suspect", "fugitive", "material witness", "missing person",
            "identify or locate", "identify", "locate"
        ]
        if any(keyword in question_lower for keyword in f2_keywords):
            logger.info("Inferred anchor prefix: §164.512(f)(2) (suspect/fugitive/witness/missing person)")
            return "§164.512(f)(2)"
        
        # Шаг 2.1.1: Проверяем law enforcement -> §164.512(f)
        if "law enforcement" in question_lower:
            logger.info("Inferred anchor prefix: §164.512(f) (law enforcement)")
            return "§164.512(f)"
        
        # Fallback: проверяем известные темы из старого маппинга
        for topic, prefix in self.TOPIC_ANCHOR_PREFIXES.items():
            if topic in question_lower:
                logger.info(f"Inferred anchor prefix: {prefix} (topic: {topic})")
                return prefix
        
        # Дефолт для law enforcement (наиболее частый кейс)
        logger.info("Using default anchor prefix: §164.512")
        return "§164.512"
    
    async def _vector_search_citation(
        self,
        question_embedding: List[float],
        doc_id: str,
        anchor_like: Optional[str],
        limit: int
    ) -> List[CandidateChunk]:
        """Vector search по atomic + hard filter по anchor prefix."""
        with self.db.cursor() as cur:
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.text_raw,
                    c.page_start,
                    c.page_end,
                    1 - (c.embedding <=> %s::vector) AS vector_score
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.part = 164
                  AND c.embedding IS NOT NULL
            """
            params = [question_embedding, doc_id]
            
            if anchor_like:
                query += " AND c.anchor LIKE %s"
                params.append(anchor_like)
            
            query += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
            params.extend([question_embedding, limit])
            
            cur.execute(query, params)
            
            candidates = []
            for row in cur.fetchall():
                candidate = CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[5],
                    page_start=row[6],
                    page_end=row[7],
                    vector_score=float(row[8]),
                    fts_score=0.0,
                    final_score=0.0
                )
                candidates.append(candidate)
            
            return candidates
    
    async def _fts_search_citation(
        self,
        fts_query: str,
        doc_id: str,
        anchor_like: Optional[str],
        limit: int
    ) -> List[CandidateChunk]:
        """FTS search внутри уже отфильтрованного anchor scope."""
        with self.db.cursor() as cur:
            if not fts_query or not fts_query.strip():
                return []
            
            try:
                query = """
                    SELECT 
                        c.chunk_id,
                        c.anchor,
                        c.section_id,
                        c.section_number,
                        c.section_title,
                        c.text_raw,
                        c.page_start,
                        c.page_end,
                        ts_rank_cd(
                            to_tsvector('english', COALESCE(c.text_raw, '')),
                            plainto_tsquery('english', %s)
                        ) AS fts_score
                    FROM chunks c
                    WHERE c.doc_id = %s
                      AND c.granularity = 'atomic'
                      AND c.part = 164
                      AND plainto_tsquery('english', %s) @@ to_tsvector('english', COALESCE(c.text_raw, ''))
                """
                params = [fts_query, doc_id, fts_query]
                
                if anchor_like:
                    query += " AND c.anchor LIKE %s"
                    params.append(anchor_like)
                
                query += " ORDER BY fts_score DESC LIMIT %s"
                params.append(limit)
                
                cur.execute(query, params)
                
                candidates = []
                for row in cur.fetchall():
                    candidate = CandidateChunk(
                        chunk_id=row[0],
                        anchor=row[1],
                        section_id=row[2],
                        section_number=row[3],
                        section_title=row[4],
                        text_raw=row[5],
                        page_start=row[6],
                        page_end=row[7],
                        vector_score=0.0,
                        fts_score=float(row[8]) if row[8] else 0.0,
                        final_score=0.0
                    )
                    candidates.append(candidate)
                
                return candidates
            except Exception as e:
                logger.warning(f"FTS поиск citation не удался: {e}")
                self.db.rollback()
                return []
    
    def _merge_and_score_citation(
        self,
        vector_candidates: List[CandidateChunk],
        fts_candidates: List[CandidateChunk]
    ) -> List[CandidateChunk]:
        """
        Объединяет vector и FTS кандидатов, нормализует и ранжирует.
        
        final_score = 0.8*vector_score + 0.2*fts_score_norm
        """
        # Нормализуем FTS scores
        normalize_fts_scores(fts_candidates)
        
        # Объединяем кандидатов
        merged_dict = merge_candidates(vector_candidates, fts_candidates)
        
        # Вычисляем final_score для всех (0.8 vector + 0.2 fts, или только vector если FTS нет)
        for candidate in merged_dict.values():
            if candidate.fts_score > 0:
                candidate.final_score = calculate_final_score(
                    candidate,
                    vector_weight=0.8,
                    fts_weight=0.2
                )
            else:
                candidate.final_score = candidate.vector_score  # Только vector если FTS нет
        
        # Сортируем по final_score
        merged_list = sorted(merged_dict.values(), key=lambda x: x.final_score, reverse=True)
        
        return merged_list
    
    async def _expand_incomplete_seed(
        self,
        seed_anchor: str,
        doc_id: str,
        limit: int = 8
    ) -> List[CandidateChunk]:
        """
        Шаг 3: Coverage completion для incomplete seed.
        
        Ищет дочерние подпункты для incomplete seed (заголовка, заканчивающегося на ":").
        Например, для §164.512(f)(1) ищет §164.512(f)(1)(i), §164.512(f)(1)(ii) и т.д.
        
        Args:
            seed_anchor: Anchor incomplete seed (например, "§164.512(f)(1)")
            doc_id: ID документа
            limit: Максимальное количество дочерних подпунктов
        
        Returns:
            Список дочерних подпунктов (CandidateChunk)
        """
        if not seed_anchor:
            return []
        
        with self.db.cursor() as cur:
            # Ищем дочерние подпункты: anchor LIKE seed_anchor || '%'
            # Исключаем сам seed_anchor
            anchor_like = f"{seed_anchor}%"
            
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.text_raw,
                    c.page_start,
                    c.page_end
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.part = 164
                  AND c.anchor LIKE %s
                  AND c.anchor <> %s
                ORDER BY c.anchor
                LIMIT %s
            """
            
            cur.execute(query, (doc_id, anchor_like, seed_anchor, limit))
            
            children = []
            for row in cur.fetchall():
                candidate = CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[5],
                    page_start=row[6],
                    page_end=row[7],
                    vector_score=0.0,  # Дочерние подпункты не имеют vector_score (они подтягиваются структурно)
                    fts_score=0.0,
                    final_score=0.0
                )
                children.append(candidate)
            
            logger.info(f"Expanded incomplete seed {seed_anchor}: found {len(children)} child subparagraphs")
            return children
    
    async def _expand_coverage(
        self,
        seeds: List[CandidateChunk],
        doc_id: str,
        anchor_like: str,
        k: int
    ) -> List[CandidateChunk]:
        """
        Coverage expansion: подтянуть соседей по section_id в пределах anchor_like.
        
        Если seed уже покрывают разные подпункты (разные anchors), можно не расширять.
        Если seed попали в один подпункт, расширить до близких 2-4 подпунктов.
        """
        if not seeds:
            return []
        
        # Проверяем, покрывают ли seeds разные подпункты
        unique_anchors = set(seed.anchor for seed in seeds if seed.anchor)
        
        # Если уже есть разные anchors, расширяем минимально
        if len(unique_anchors) >= 3:
            return []
        
        # Получаем все atomic чанки в пределах anchor_like, отсортированные по anchor
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.text_raw,
                    c.page_start,
                    c.page_end
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.part = 164
                  AND c.anchor LIKE %s
                ORDER BY c.anchor
                LIMIT %s
            """, (doc_id, anchor_like, k * 2))
            
            all_chunks = []
            seen_chunk_ids = {seed.chunk_id for seed in seeds}
            
            for row in cur.fetchall():
                if row[0] in seen_chunk_ids:
                    continue
                
                candidate = CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[5],
                    page_start=row[6],
                    page_end=row[7],
                    vector_score=0.0,
                    fts_score=0.0,
                    final_score=0.0
                )
                all_chunks.append(candidate)
            
            # Шаг 2.4: Берем ближайших соседей к seeds (по порядку anchor), но только внутри prefix
            expanded = []
            seed_anchors = [s.anchor for s in seeds if s.anchor]
            
            for candidate in all_chunks:
                if candidate.anchor:
                    # Шаг 2.4: Фильтруем только те, что начинаются с prefix (дополнительная проверка)
                    if not candidate.anchor.startswith(prefix):
                        continue
                    
                    # Проверяем, является ли этот chunk соседом какого-либо seed
                    for seed_anchor in seed_anchors:
                        # Простая эвристика: если anchor близок по алфавиту/порядку
                        if self._are_anchors_nearby(candidate.anchor, seed_anchor):
                            expanded.append(candidate)
                            break
                
                if len(expanded) >= 4:  # Максимум 4 соседа
                    break
            
            return expanded
    
    async def _get_sibling_chunks(
        self,
        seed: CandidateChunk,
        doc_id: str,
        anchor_like: Optional[str],
        limit: int = 12
    ) -> List[CandidateChunk]:
        """
        Шаг 4 Fallback B: Получает sibling chunks из той же секции.
        
        Если list introducer не имеет дочерних подпунктов, возвращает соседние чанки
        из той же секции в пределах anchor_like.
        
        Args:
            seed: Seed chunk (list introducer)
            doc_id: ID документа
            anchor_like: Anchor prefix для фильтрации (например, "§164.512(f)%")
            limit: Максимальное количество sibling chunks
        
        Returns:
            Список sibling chunks (CandidateChunk)
        """
        if not seed.section_id:
            return []
        
        with self.db.cursor() as cur:
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.text_raw,
                    c.page_start,
                    c.page_end
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.section_id = %s
                  AND c.chunk_id <> %s
            """
            params = [doc_id, seed.section_id, seed.chunk_id]
            
            # Дополнительно ограничиваем по scope: anchor LIKE anchor_like
            if anchor_like:
                query += " AND c.anchor LIKE %s"
                params.append(anchor_like)
            
            query += " ORDER BY c.anchor LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            
            siblings = []
            for row in cur.fetchall():
                candidate = CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[5],
                    page_start=row[6],
                    page_end=row[7],
                    vector_score=0.0,
                    fts_score=0.0,
                    final_score=0.0
                )
                siblings.append(candidate)
            
            logger.info(f"Found {len(siblings)} sibling chunks for seed {seed.anchor} in section {seed.section_id}")
            return siblings
    
    def _are_anchors_nearby(self, anchor1: str, anchor2: str) -> bool:
        """
        Проверяет, являются ли два anchor соседями (в пределах одной секции).
        
        Простая эвристика: если anchor имеют общий префикс и отличаются только подпунктом.
        """
        # Извлекаем базовый префикс (например, "§164.512")
        prefix1 = re.match(r'^([^:]+)', anchor1)
        prefix2 = re.match(r'^([^:]+)', anchor2)
        
        if not prefix1 or not prefix2:
            return False
        
        if prefix1.group(1) != prefix2.group(1):
            return False
        
        # Проверяем, что они в одной секции (до двоеточия или скобки)
        section1 = re.match(r'^([^:(]+)', anchor1)
        section2 = re.match(r'^([^:(]+)', anchor2)
        
        if section1 and section2:
            return section1.group(1) == section2.group(1)
        
        return False
    
    def _dedup_and_select_citation(
        self,
        candidates: List[CandidateChunk],
        k: int
    ) -> List[CandidateChunk]:
        """
        Дедупликация и выбор top-k с сортировкой по anchor.
        """
        # Дедуп по chunk_id
        seen = set()
        deduped = []
        for candidate in candidates:
            if candidate.chunk_id not in seen:
                deduped.append(candidate)
                seen.add(candidate.chunk_id)
        
        # Сортировка по anchor (чтобы подпункты шли в порядке документа)
        deduped.sort(key=lambda x: x.anchor or "")
        
        # Ограничиваем k
        return deduped[:k]
