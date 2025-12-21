"""Ретривер для вопросов типа 'definition' - поиск определений терминов."""

from typing import List, Optional, Dict, Any
import logging
import re

from .base import BaseRetriever, CandidateChunk
from .utils import get_db_connection, normalize_fts_scores, merge_candidates, calculate_final_score
from embeddings import get_embedding_client

logger = logging.getLogger(__name__)


class DefinitionRetriever(BaseRetriever):
    """
    Ретривер для вопросов типа 'definition' - поиск определений терминов.
    
    Возвращает точные определения из нормативного текста (1-2 атомарных подпараграфа).
    Использует двухступенчатый поиск: таблица definitions (если есть) + fallback по atomic chunks.
    """
    
    def __init__(self, db_connection=None):
        """
        Инициализация ретривера определений.
        
        Args:
            db_connection: Подключение к PostgreSQL (если None, создается новое)
        """
        self.db = db_connection or get_db_connection()
        self.embedding_client = get_embedding_client()
    
    async def retrieve(
        self,
        question_embedding: List[float],
        max_results: int = 1,
        question: Optional[str] = None,
        doc_id: Optional[str] = None,
        k: int = 1,
        strict_quote: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Поиск определений терминов.
        
        Args:
            question_embedding: Эмбеддинг вопроса (размерность 4096)
            max_results: Максимальное количество результатов (по умолчанию 1, максимум 2)
            question: Текст вопроса
            doc_id: ID документа (по умолчанию 'hipaa-reg-2013-03-26')
            k: Количество результатов (по умолчанию 1, иногда 2 если определение разбито)
            strict_quote: Требовать точную цитату (по умолчанию True)
        
        Returns:
            Список словарей с информацией об определениях
        """
        doc_id = doc_id or "hipaa-reg-2013-03-26"
        question = question or ""
        k = min(k, max_results, 2)  # Максимум 2 определения
        
        logger.info(f"DefinitionRetriever (from new module): question='{question[:50]}...', k={k}")
        
        # Step 0: Извлечение термина из вопроса
        term = self._extract_term(question)
        logger.info(f"Извлеченный термин: {term if term else 'не найден'}")
        
        # Step 1: Поиск в таблице definitions (если есть)
        # Пока пропускаем, так как таблицы definitions нет в схеме
        # definition_results = await self._search_definitions_table(term, doc_id, k)
        # if definition_results:
        #     return definition_results
        
        # Step 2: Fallback - semantic + lexical поиск по atomic chunks
        candidates = await self._search_atomic_chunks(
            question_embedding=question_embedding,
            question=question,
            term=term,
            doc_id=doc_id,
            limit=k * 5  # Берем больше кандидатов для валидации
        )
        
        if not candidates:
            logger.warning("Не найдено кандидатов для определения")
            return []
        
        # Step 3: Пост-проверки качества и выбор top-k
        validated_results = []
        for candidate in candidates:
            if self._validate_result(candidate, term):
                validated_results.append(candidate)
                if len(validated_results) >= k:
                    break
        
        # Если после валидации ничего не осталось, берем лучший результат
        if not validated_results and candidates:
            logger.warning("Валидация провалилась, берем лучший результат")
            validated_results = [candidates[0]]
        
        # Конвертируем в формат API
        results = []
        for candidate in validated_results[:k]:
            results.append({
                "chunk_id": candidate.chunk_id,
                "anchor": candidate.anchor,
                "section_id": candidate.section_id,
                "section_number": candidate.section_number,
                "section_title": candidate.section_title,
                "text_raw": candidate.text_raw,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "scores": {
                    "def_table_score": 0.0,  # Не использовали таблицу definitions
                    "vector_score": candidate.vector_score,
                    "fts_score": candidate.fts_score,
                    "final_score": candidate.final_score
                },
                "term": term or "",
                "explanation": f"fallback semantic search" + (f" (term: {term})" if term else "")
            })
        
        logger.info(f"Найдено определений: {len(results)}")
        return results
    
    def _extract_term(self, question: str) -> Optional[str]:
        """
        Извлекает термин из вопроса.
        
        Эвристики:
        - Если есть кавычки - берем содержимое
        - Шаблоны: "what does X mean", "define X", "what is a/an X"
        - Нормализация: lower, trim, убрать финальные знаки препинания
        """
        if not question:
            return None
        
        # Проверка кавычек
        quoted = re.search(r'["\']([^"\']+)["\']', question)
        if quoted:
            term = quoted.group(1).strip().lower()
            # Убираем финальные знаки препинания
            term = re.sub(r'[?.,:;]+$', '', term)
            if term:
                return term
        
        # Шаблоны
        patterns = [
            r"what does ['\"]?([^?'\"]+)['\"]? mean",
            r"define ['\"]?([^?'\"]+)['\"]?",
            r"what is (?:a|an) ['\"]?([^?'\"]+)['\"]?",
            r"definition of ['\"]?([^?'\"]+)['\"]?",
            r"['\"]?([^?'\"]+)['\"]? (?:means|refers to)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                term = match.group(1).strip()
                # Убираем финальные знаки препинания
                term = re.sub(r'[?.,:;]+$', '', term)
                # Убираем артикли в начале
                term = re.sub(r'^(a|an|the)\s+', '', term)
                if term and len(term) > 2:  # Минимальная длина термина
                    return term
        
        return None
    
    async def _search_definitions_table(
        self,
        term: Optional[str],
        doc_id: str,
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Поиск в таблице definitions (если заполнена).
        
        TODO: Реализовать, когда таблица definitions будет создана.
        """
        # Проверяем, существует ли таблица
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'definitions'
                )
            """)
            table_exists = cur.fetchone()[0]
            
            if not table_exists or not term:
                return []
            
            # 1A) Exact match
            cur.execute("""
                SELECT 
                    d.term,
                    d.term_surface,
                    d.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.text_raw,
                    c.page_start,
                    c.page_end,
                    1.0 AS def_table_score
                FROM definitions d
                JOIN chunks c ON c.chunk_id = d.chunk_id
                WHERE d.doc_id = %s
                  AND lower(d.term) = lower(%s)
                ORDER BY c.anchor
                LIMIT %s
            """, (doc_id, term, k))
            
            rows = cur.fetchall()
            if rows:
                results = []
                for row in rows:
                    results.append({
                        "chunk_id": row[2],
                        "anchor": row[3],
                        "section_id": row[4],
                        "section_number": row[5],
                        "section_title": row[6],
                        "text_raw": row[7],
                        "page_start": row[8],
                        "page_end": row[9],
                        "scores": {
                            "def_table_score": float(row[10]),
                            "vector_score": 0.0,
                            "fts_score": 0.0,
                            "final_score": float(row[10])
                        },
                        "term": row[0],
                        "explanation": "matched from definitions table (exact)"
                    })
                return results
            
            # 1B) Fuzzy match (trigram) если exact не нашлось
            try:
                cur.execute("""
                    SELECT 
                        d.term,
                        d.term_surface,
                        d.chunk_id,
                        c.anchor,
                        c.section_id,
                        c.section_number,
                        c.section_title,
                        c.text_raw,
                        c.page_start,
                        c.page_end,
                        similarity(lower(d.term), lower(%s)) AS def_table_score
                    FROM definitions d
                    JOIN chunks c ON c.chunk_id = d.chunk_id
                    WHERE d.doc_id = %s
                      AND similarity(lower(d.term), lower(%s)) >= 0.65
                    ORDER BY def_table_score DESC
                    LIMIT %s
                """, (term, doc_id, term, k))
                
                rows = cur.fetchall()
                if rows:
                    results = []
                    for row in rows:
                        results.append({
                            "chunk_id": row[2],
                            "anchor": row[3],
                            "section_id": row[4],
                            "section_number": row[5],
                            "section_title": row[6],
                            "text_raw": row[7],
                            "page_start": row[8],
                            "page_end": row[9],
                            "scores": {
                                "def_table_score": float(row[10]),
                                "vector_score": 0.0,
                                "fts_score": 0.0,
                                "final_score": float(row[10])
                            },
                            "term": row[0],
                            "explanation": f"matched from definitions table (fuzzy, score={row[10]:.2f})"
                        })
                    return results
            except Exception as e:
                logger.warning(f"Fuzzy match в definitions table не удался: {e}")
                self.db.rollback()
        
        return []
    
    async def _search_atomic_chunks(
        self,
        question_embedding: List[float],
        question: str,
        term: Optional[str],
        doc_id: str,
        limit: int
    ) -> List[CandidateChunk]:
        """
        Fallback поиск по atomic chunks с фильтрами.
        
        Фильтры:
        - granularity = 'atomic'
        - chunk_kind = 'definition'
        - FTS по term (если есть) или по question
        """
        # FTS query: используем term если есть, иначе question
        fts_query = term if term else question
        
        # 2A) FTS candidates
        fts_candidates = await self._fts_search_atomic(fts_query, doc_id, limit)
        logger.info(f"FTS кандидатов (atomic, definition): {len(fts_candidates)}")
        
        # 2B) Vector candidates
        vector_candidates = await self._vector_search_atomic(question_embedding, doc_id, limit)
        logger.info(f"Vector кандидатов (atomic, definition): {len(vector_candidates)}")
        
        # 2C) Merge + rerank
        merged = self._merge_and_rerank_definition(fts_candidates, vector_candidates, term is not None)
        
        return merged
    
    async def _fts_search_atomic(
        self,
        fts_query: str,
        doc_id: str,
        limit: int
    ) -> List[CandidateChunk]:
        """FTS поиск по atomic chunks с фильтром chunk_kind='definition'."""
        with self.db.cursor() as cur:
            if fts_query and fts_query.strip():
                try:
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
                            ts_rank_cd(
                                to_tsvector('english', COALESCE(c.text_raw, '')),
                                plainto_tsquery('english', %s)
                            ) AS fts_score
                        FROM chunks c
                        WHERE c.doc_id = %s
                          AND c.granularity = 'atomic'
                          AND c.chunk_kind = 'definition'
                          AND c.embedding IS NOT NULL
                          AND plainto_tsquery('english', %s) @@ to_tsvector('english', COALESCE(c.text_raw, ''))
                        ORDER BY fts_score DESC
                        LIMIT %s
                    """, (fts_query, doc_id, fts_query, limit))
                    
                    rows = cur.fetchall()
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
                            vector_score=0.0,
                            fts_score=float(row[8]) if row[8] else 0.0,
                            final_score=0.0
                        ))
                    return candidates
                except Exception as e:
                    logger.warning(f"FTS поиск atomic chunks не удался: {e}")
                    self.db.rollback()
            
            return []
    
    async def _vector_search_atomic(
        self,
        question_embedding: List[float],
        doc_id: str,
        limit: int
    ) -> List[CandidateChunk]:
        """Vector similarity поиск по atomic chunks с фильтром chunk_kind='definition'."""
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
                    c.page_end,
                    1 - (c.embedding <=> %s::vector) AS vector_score
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.chunk_kind = 'definition'
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """, (question_embedding, doc_id, question_embedding, limit))
            
            candidates = []
            for row in cur.fetchall():
                candidates.append(CandidateChunk(
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
                ))
            
            return candidates
    
    def _merge_and_rerank_definition(
        self,
        fts_candidates: List[CandidateChunk],
        vector_candidates: List[CandidateChunk],
        has_term: bool
    ) -> List[CandidateChunk]:
        """
        Объединяет результаты FTS и vector, нормализует и ранжирует для определений.
        
        Если term извлечен:
            final_score = 0.6*vector_score + 0.4*fts_score_norm
        Если term нет:
            final_score = 0.8*vector_score + 0.2*fts_score_norm
        """
        # Нормализуем FTS scores
        normalize_fts_scores(fts_candidates)
        
        # Объединяем кандидатов
        merged_dict = merge_candidates(vector_candidates, fts_candidates)
        
        # Вычисляем final_score для всех (с учетом has_term)
        for candidate in merged_dict.values():
            if has_term:
                candidate.final_score = calculate_final_score(
                    candidate,
                    vector_weight=0.6,
                    fts_weight=0.4
                )
            else:
                candidate.final_score = calculate_final_score(
                    candidate,
                    vector_weight=0.8,
                    fts_weight=0.2
                )
        
        # Сортируем по final_score
        merged_list = sorted(merged_dict.values(), key=lambda x: x.final_score, reverse=True)
        
        return merged_list
    
    def _validate_result(self, candidate: CandidateChunk, term: Optional[str]) -> bool:
        """
        Пост-проверки качества результата для определения.
        
        Проверки:
        - text_raw содержит паттерн определения (начинается с "<TERM> means" или содержит "means" / "refers to")
        - или это внутри секции "Definitions"
        - anchor не пустой
        """
        if not candidate.anchor or not candidate.anchor.strip():
            logger.debug(f"Валидация провалена: пустой anchor для {candidate.chunk_id}")
            return False
        
        text_lower = candidate.text_raw.lower()
        
        # Проверка паттернов определения
        has_definition_pattern = (
            "means" in text_lower or
            "refers to" in text_lower or
            "is defined" in text_lower or
            text_lower.startswith(f"{term.lower()} means") if term else False
        )
        
        # Проверка секции Definitions
        is_definitions_section = (
            "definition" in candidate.section_title.lower() or
            "definition" in candidate.section_number.lower()
        )
        
        if has_definition_pattern or is_definitions_section:
            return True
        
        # Если термин есть, проверяем его наличие в тексте
        if term and term.lower() in text_lower:
            return True
        
        logger.debug(f"Валидация провалена: нет паттернов определения для {candidate.chunk_id}")
        return False
