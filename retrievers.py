"""Ретриверы для поиска релевантных чанков в зависимости от категории вопроса."""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import re
import psycopg
from pathlib import Path

from embeddings import get_embedding_client

logger = logging.getLogger(__name__)


@dataclass
class CandidateChunk:
    """Кандидат-чанк с оценками."""
    chunk_id: str
    anchor: Optional[str]
    section_id: str
    section_number: str
    section_title: str
    text_raw: str
    page_start: Optional[int]
    page_end: Optional[int]
    vector_score: float
    fts_score: float
    final_score: float


@dataclass
class NavigationHit:
    """Результат навигационного поиска - структурный элемент документа."""
    part: Optional[int]
    subpart: Optional[str]
    section_id: str
    section_number: str
    section_title: str
    anchor: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    scores: Dict[str, float]  # rule_score, title_score, final_score
    explanation: str


class BaseRetriever(ABC):
    """Базовый класс для ретриверов."""
    
    @abstractmethod
    async def retrieve(
        self,
        question_embedding: List[float],
        max_results: int = 5,
        **kwargs
    ) -> List[dict]:
        """
        Поиск релевантных чанков.
        
        Args:
            question_embedding: Эмбеддинг вопроса
            max_results: Максимальное количество результатов
            **kwargs: Дополнительные параметры для конкретного ретривера
        
        Returns:
            Список словарей с информацией о чанках
        """
        pass


def get_db_connection():
    """Создает подключение к базе данных."""
    user = Path.home().name
    return psycopg.connect(
        host="localhost",
        dbname="raft",
        user=user,
    )


class SemanticRetriever(BaseRetriever):
    """Семантический ретривер - поиск по косинусному сходству эмбеддингов."""
    
    def __init__(self, db_connection=None):
        """
        Инициализация семантического ретривера.
        
        Args:
            db_connection: Подключение к PostgreSQL (если None, создается новое)
        """
        self.db = db_connection or get_db_connection()
    
    async def retrieve(
        self,
        question_embedding: List[float],
        max_results: int = 5,
        **kwargs
    ) -> List[dict]:
        """
        Поиск по семантическому сходству.
        
        TODO: Реализовать поиск в PostgreSQL с использованием pgvector.
        """
        # Заглушка - будет реализовано позже
        logger.warning("SemanticRetriever.retrieve: заглушка")
        return []


class OverviewPurposeRetriever(BaseRetriever):
    """
    Ретривер для вопросов типа 'overview / purpose'.
    
    Возвращает 1-2 крупных section-level фрагмента, описывающих общий смысл/назначение.
    Использует гибридный поиск: FTS + vector similarity.
    """
    
    def __init__(self, db_connection=None):
        """
        Инициализация ретривера.
        
        Args:
            db_connection: Подключение к PostgreSQL (если None, создается новое)
        """
        self.db = db_connection or get_db_connection()
        self.embedding_client = get_embedding_client()
    
    async def retrieve(
        self,
        question_embedding: List[float],
        max_results: int = 2,
        part: Optional[int] = None,
        doc_id: Optional[str] = None,
        question: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Поиск section-level чанков для overview/purpose.
        
        Args:
            question_embedding: Эмбеддинг вопроса (размерность 4096)
            max_results: Количество результатов (по умолчанию 2)
            part: Номер части (например, 160)
            doc_id: ID документа (по умолчанию 'hipaa-reg-2013-03-26')
            question: Текст вопроса (для FTS поиска)
        
        Returns:
            Список словарей с информацией о чанках
        """
        doc_id = doc_id or "hipaa-reg-2013-03-26"
        
        # Если part не указан, пытаемся извлечь из вопроса или используем дефолт
        if part is None:
            part = 160  # Дефолт для HIPAA
        
        logger.info(f"OverviewPurposeRetriever: part={part}, max_results={max_results}")
        
        # Шаг 2A: FTS candidates
        fts_candidates = await self._fts_search(question or "", part, doc_id, max_results * 3)
        logger.info(f"FTS кандидатов: {len(fts_candidates)}")
        
        # Шаг 2B: Vector candidates
        vector_candidates = await self._vector_search(question_embedding, part, doc_id, max_results * 3)
        logger.info(f"Vector кандидатов: {len(vector_candidates)}")
        
        # Шаг 3: Merge + rerank
        merged = self._merge_and_rerank(fts_candidates, vector_candidates)
        
        # Шаг 4: Dedup + select top-k
        final_results = self._dedup_and_select(merged, max_results)
        
        logger.info(f"Финальных результатов: {len(final_results)}")
        
        # Конвертируем в словари для API
        return [self._candidate_to_dict(c) for c in final_results]
    
    async def _fts_search(
        self,
        question: str,
        part: int,
        doc_id: str,
        limit: int
    ) -> List[CandidateChunk]:
        """FTS поиск (lexical)."""
        with self.db.cursor() as cur:
            if question and question.strip():
                # Используем plainto_tsquery для полнотекстового поиска
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
                                to_tsvector('english', COALESCE(c.section_title, '') || ' ' || COALESCE(c.text_raw, '')),
                                plainto_tsquery('english', %s)
                            ) AS fts_score
                        FROM chunks c
                        WHERE c.doc_id = %s
                          AND c.part = %s
                          AND c.granularity = 'section'
                          AND c.embedding IS NOT NULL
                          AND plainto_tsquery('english', %s) @@ to_tsvector('english', COALESCE(c.section_title, '') || ' ' || COALESCE(c.text_raw, ''))
                        ORDER BY fts_score DESC
                        LIMIT %s
                    """, (question, doc_id, part, question, limit))
                    
                    rows = cur.fetchall()
                    if rows:
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
                    logger.warning(f"FTS поиск не удался, используем fallback: {e}")
            
            # Fallback: упрощенный вариант с приоритетами по ключевым словам
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
                        WHEN c.section_title ILIKE '%%purpose%%' OR c.section_title ILIKE '%%basis%%' OR c.section_title ILIKE '%%scope%%'
                        THEN 1.0
                        WHEN c.chunk_kind IN ('scope', 'other')
                        THEN 0.8
                        ELSE 0.5
                    END AS fts_score
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.part = %s
                  AND c.granularity = 'section'
                  AND c.embedding IS NOT NULL
                ORDER BY fts_score DESC, c.section_number
                LIMIT %s
            """, (doc_id, part, limit))
            
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
                    vector_score=0.0,  # Будет заполнено при merge
                    fts_score=float(row[8]),
                    final_score=0.0
                ))
            
            return candidates
    
    async def _vector_search(
        self,
        question_embedding: List[float],
        part: int,
        doc_id: str,
        limit: int
    ) -> List[CandidateChunk]:
        """Vector similarity поиск (semantic)."""
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
                  AND c.part = %s
                  AND c.granularity = 'section'
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """, (question_embedding, doc_id, part, question_embedding, limit))
            
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
                    fts_score=0.0,  # Будет заполнено при merge
                    final_score=0.0
                ))
            
            return candidates
    
    def _merge_and_rerank(
        self,
        fts_candidates: List[CandidateChunk],
        vector_candidates: List[CandidateChunk]
    ) -> List[CandidateChunk]:
        """Объединяет результаты FTS и vector, нормализует и ранжирует."""
        # Создаем словарь по chunk_id для объединения
        merged_dict: Dict[str, CandidateChunk] = {}
        
        # Нормализация FTS scores
        if fts_candidates:
            max_fts = max(c.fts_score for c in fts_candidates) if fts_candidates else 1.0
            if max_fts > 0:
                for c in fts_candidates:
                    c.fts_score = c.fts_score / max_fts
        
        # Добавляем FTS кандидатов
        for candidate in fts_candidates:
            merged_dict[candidate.chunk_id] = candidate
        
        # Объединяем с vector кандидатами
        for candidate in vector_candidates:
            if candidate.chunk_id in merged_dict:
                # Объединяем оценки
                existing = merged_dict[candidate.chunk_id]
                existing.vector_score = candidate.vector_score
                # FTS score уже есть
            else:
                # Добавляем новый
                merged_dict[candidate.chunk_id] = candidate
        
        # Вычисляем final_score для всех
        for candidate in merged_dict.values():
            candidate.final_score = 0.7 * candidate.vector_score + 0.3 * candidate.fts_score
        
        # Сортируем по final_score
        merged_list = sorted(merged_dict.values(), key=lambda x: x.final_score, reverse=True)
        
        return merged_list
    
    def _dedup_and_select(
        self,
        candidates: List[CandidateChunk],
        max_results: int
    ) -> List[CandidateChunk]:
        """Дедупликация по section_id и выбор top-k."""
        seen_sections = set()
        result = []
        
        for candidate in candidates:
            if candidate.section_id not in seen_sections:
                result.append(candidate)
                seen_sections.add(candidate.section_id)
                if len(result) >= max_results:
                    break
        
        return result
    
    def _candidate_to_dict(self, candidate: CandidateChunk) -> Dict[str, Any]:
        """Конвертирует CandidateChunk в словарь для API."""
        return {
            "chunk_id": candidate.chunk_id,
            "anchor": candidate.anchor,
            "section_id": candidate.section_id,
            "section_number": candidate.section_number,
            "section_title": candidate.section_title,
            "text_raw": candidate.text_raw,
            "page_start": candidate.page_start,
            "page_end": candidate.page_end,
            "scores": {
                "vector_score": candidate.vector_score,
                "fts_score": candidate.fts_score,
                "final_score": candidate.final_score
            }
        }


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
        
        logger.info(f"DefinitionRetriever: question='{question[:50]}...', k={k}")
        
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
        import re
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
        # Нормализация FTS scores
        if fts_candidates:
            max_fts = max(c.fts_score for c in fts_candidates) if fts_candidates else 1.0
            if max_fts > 0:
                for c in fts_candidates:
                    c.fts_score = c.fts_score / max_fts
        
        # Создаем словарь по chunk_id для объединения
        merged_dict: Dict[str, CandidateChunk] = {}
        
        # Добавляем FTS кандидатов
        for candidate in fts_candidates:
            merged_dict[candidate.chunk_id] = candidate
        
        # Объединяем с vector кандидатами
        for candidate in vector_candidates:
            if candidate.chunk_id in merged_dict:
                # Объединяем оценки
                existing = merged_dict[candidate.chunk_id]
                existing.vector_score = candidate.vector_score
            else:
                # Добавляем новый
                merged_dict[candidate.chunk_id] = candidate
        
        # Вычисляем final_score для всех
        for candidate in merged_dict.values():
            if has_term:
                candidate.final_score = 0.6 * candidate.vector_score + 0.4 * candidate.fts_score
            else:
                candidate.final_score = 0.8 * candidate.vector_score + 0.2 * candidate.fts_score
        
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


class ScopeRetriever(BaseRetriever):
    """
    Ретривер для вопросов типа 'scope / applicability'.
    
    Возвращает набор релевантных подпунктов (обычно 2-6 atomic чанков) с сохранением anchor'ов.
    Использует seed + sibling expansion для сбора полного списка категорий.
    """
    
    # Буст-слова для FTS query
    APPLICABILITY_BOOST_WORDS = [
        "apply", "applicability", "applies", "entity", "entities",
        "covered", "regulated", "health plan", "clearinghouse",
        "provider", "business associate"
    ]
    
    def __init__(self, db_connection=None):
        """
        Инициализация ретривера применимости.
        
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
        part: Optional[int] = None,
        k: int = 6,
        expand_with_siblings: bool = True,
        seed_k: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Поиск информации об области применения.
        
        Args:
            question_embedding: Эмбеддинг вопроса (размерность 4096)
            max_results: Максимальное количество результатов (по умолчанию 6)
            question: Текст вопроса
            doc_id: ID документа (по умолчанию 'hipaa-reg-2013-03-26')
            part: Номер части (например, 160 или 164)
            k: Количество atomic пунктов для возврата (по умолчанию 6)
            expand_with_siblings: Расширять соседями (по умолчанию True)
            seed_k: Количество seed-чанков перед расширением (по умолчанию 3)
        
        Returns:
            Список словарей с информацией о применимости
        """
        doc_id = doc_id or "hipaa-reg-2013-03-26"
        question = question or ""
        k = min(k, max_results, 8)  # Максимум 8 пунктов
        
        logger.info(f"ScopeRetriever: question='{question[:50]}...', k={k}, seed_k={seed_k}, expand={expand_with_siblings}")
        
        # Step 0: Подготовка FTS query (усиление)
        fts_query = self._build_fts_query(question)
        logger.info(f"FTS query (усиленный): {fts_query[:100]}...")
        
        # Step 1: Seed retrieval (hybrid, atomic)
        seeds = await self._get_seeds(
            question_embedding=question_embedding,
            fts_query=fts_query,
            doc_id=doc_id,
            part=part,
            limit=50  # Берем больше для дедупликации
        )
        
        if not seeds:
            logger.warning("Не найдено seed-чанков")
            return []
        
        # Выбираем seed_k лучших с дедупом по section_id
        selected_seeds = self._select_seeds(seeds, seed_k)
        logger.info(f"Выбрано seed-чанков: {len(selected_seeds)}")
        
        # Step 2: Sibling expansion (если включено)
        all_results = list(selected_seeds)
        
        if expand_with_siblings:
            siblings = await self._expand_siblings(selected_seeds, doc_id)
            logger.info(f"Найдено sibling-чанков: {len(siblings)}")
            
            # Объединяем seeds и siblings
            all_results.extend(siblings)
        
        # Step 3: Dedup + selection
        final_results = self._dedup_and_select(all_results, k)
        
        logger.info(f"Финальных результатов: {len(final_results)}")
        
        # Конвертируем в формат API
        results = []
        for candidate in final_results:
            results.append({
                "chunk_id": candidate.chunk_id,
                "anchor": candidate.anchor,
                "section_id": candidate.section_id,
                "section_number": candidate.section_number,
                "section_title": candidate.section_title,
                "paragraph_path": getattr(candidate, 'paragraph_path', None),
                "text_raw": candidate.text_raw,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "scores": {
                    "vector_score": candidate.vector_score,
                    "fts_score": candidate.fts_score,
                    "final_score": candidate.final_score
                },
                "explanation": getattr(candidate, 'explanation', 'seed from semantic search')
            })
        
        return results
    
    def _build_fts_query(self, question: str) -> str:
        """
        Формирует усиленный FTS query из вопроса + буст-слова.
        """
        if not question:
            return " ".join(self.APPLICABILITY_BOOST_WORDS)
        
        # Добавляем буст-слова к вопросу
        boost_text = " ".join(self.APPLICABILITY_BOOST_WORDS)
        return f"{question} {boost_text}"
    
    async def _get_seeds(
        self,
        question_embedding: List[float],
        fts_query: str,
        doc_id: str,
        part: Optional[int],
        limit: int
    ) -> List[CandidateChunk]:
        """
        Получает seed-чанки через hybrid поиск (vector + FTS).
        """
        # 1A) Vector seeds
        vector_seeds = await self._vector_search_seeds(question_embedding, doc_id, part, limit)
        logger.info(f"Vector seeds: {len(vector_seeds)}")
        
        # 1B) FTS seeds (усиленные)
        fts_seeds = await self._fts_search_seeds(fts_query, doc_id, part, limit)
        logger.info(f"FTS seeds: {len(fts_seeds)}")
        
        # 1C) Merge + scoring
        merged = self._merge_and_score_seeds(vector_seeds, fts_seeds)
        
        return merged
    
    async def _vector_search_seeds(
        self,
        question_embedding: List[float],
        doc_id: str,
        part: Optional[int],
        limit: int
    ) -> List[CandidateChunk]:
        """Vector similarity поиск для seeds."""
        with self.db.cursor() as cur:
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.paragraph_path,
                    c.parent_chunk_id,
                    c.text_raw,
                    c.page_start,
                    c.page_end,
                    1 - (c.embedding <=> %s::vector) AS vector_score
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.embedding IS NOT NULL
            """
            params = [question_embedding, doc_id]
            
            if part is not None:
                query += " AND c.part = %s"
                params.append(part)
            
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
                    text_raw=row[7],
                    page_start=row[8],
                    page_end=row[9],
                    vector_score=float(row[10]),
                    fts_score=0.0,
                    final_score=0.0
                )
                # Сохраняем дополнительные поля
                candidate.paragraph_path = row[5]
                candidate.parent_chunk_id = row[6]
                candidates.append(candidate)
            
            return candidates
    
    async def _fts_search_seeds(
        self,
        fts_query: str,
        doc_id: str,
        part: Optional[int],
        limit: int
    ) -> List[CandidateChunk]:
        """FTS поиск для seeds (усиленный)."""
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
                        c.paragraph_path,
                        c.parent_chunk_id,
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
                      AND plainto_tsquery('english', %s) @@ to_tsvector('english', COALESCE(c.text_raw, ''))
                """
                params = [fts_query, doc_id, fts_query]
                
                if part is not None:
                    query += " AND c.part = %s"
                    params.append(part)
                
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
                        text_raw=row[7],
                        page_start=row[8],
                        page_end=row[9],
                        vector_score=0.0,
                        fts_score=float(row[10]) if row[10] else 0.0,
                        final_score=0.0
                    )
                    # Сохраняем дополнительные поля
                    candidate.paragraph_path = row[5]
                    candidate.parent_chunk_id = row[6]
                    candidates.append(candidate)
                
                return candidates
            except Exception as e:
                logger.warning(f"FTS поиск seeds не удался: {e}")
                self.db.rollback()
                return []
    
    def _merge_and_score_seeds(
        self,
        vector_seeds: List[CandidateChunk],
        fts_seeds: List[CandidateChunk]
    ) -> List[CandidateChunk]:
        """
        Объединяет vector и FTS seeds, нормализует и ранжирует.
        
        final_score = 0.7*vector_score + 0.3*fts_score_norm
        """
        # Нормализация FTS scores
        if fts_seeds:
            max_fts = max(c.fts_score for c in fts_seeds) if fts_seeds else 1.0
            if max_fts > 0:
                for c in fts_seeds:
                    c.fts_score = c.fts_score / max_fts
        
        # Создаем словарь по chunk_id для объединения
        merged_dict: Dict[str, CandidateChunk] = {}
        
        # Добавляем vector seeds
        for candidate in vector_seeds:
            merged_dict[candidate.chunk_id] = candidate
        
        # Объединяем с FTS seeds
        for candidate in fts_seeds:
            if candidate.chunk_id in merged_dict:
                # Объединяем оценки
                existing = merged_dict[candidate.chunk_id]
                existing.fts_score = candidate.fts_score
            else:
                # Добавляем новый
                merged_dict[candidate.chunk_id] = candidate
        
        # Вычисляем final_score для всех
        for candidate in merged_dict.values():
            candidate.final_score = 0.7 * candidate.vector_score + 0.3 * candidate.fts_score
        
        # Сортируем по final_score
        merged_list = sorted(merged_dict.values(), key=lambda x: x.final_score, reverse=True)
        
        return merged_list
    
    def _select_seeds(self, seeds: List[CandidateChunk], seed_k: int) -> List[CandidateChunk]:
        """
        Выбирает seed_k лучших seeds с дедупом по section_id.
        """
        seen_sections = set()
        selected = []
        
        for seed in seeds:
            if seed.section_id not in seen_sections:
                selected.append(seed)
                seen_sections.add(seed.section_id)
                if len(selected) >= seed_k:
                    break
        
        return selected
    
    async def _expand_siblings(
        self,
        seeds: List[CandidateChunk],
        doc_id: str
    ) -> List[CandidateChunk]:
        """
        Расширяет seeds соседями (siblings) по parent_chunk_id или section_id.
        """
        siblings = []
        seen_chunk_ids = {seed.chunk_id for seed in seeds}
        
        for seed in seeds:
            seed_siblings = []
            
            # 2A) Expansion by parent_chunk_id (предпочтительно)
            if hasattr(seed, 'parent_chunk_id') and seed.parent_chunk_id:
                seed_siblings = await self._get_siblings_by_parent(seed.parent_chunk_id, doc_id, seen_chunk_ids)
                if seed_siblings:
                    logger.debug(f"Найдено {len(seed_siblings)} siblings по parent_chunk_id для {seed.chunk_id}")
            
            # 2B) Fallback expansion by section_id
            if not seed_siblings:
                seed_siblings = await self._get_siblings_by_section(seed.section_id, doc_id, seen_chunk_ids)
                if seed_siblings:
                    logger.debug(f"Найдено {len(seed_siblings)} siblings по section_id для {seed.chunk_id}")
            
            # Присваиваем оценку siblings (чуть ниже seed)
            for sibling in seed_siblings:
                sibling.vector_score = seed.vector_score * 0.95
                sibling.fts_score = seed.fts_score * 0.95
                sibling.final_score = seed.final_score * 0.95
                sibling.explanation = "sibling expansion"
                seen_chunk_ids.add(sibling.chunk_id)
            
            siblings.extend(seed_siblings)
        
        return siblings
    
    async def _get_siblings_by_parent(
        self,
        parent_chunk_id: str,
        doc_id: str,
        seen_chunk_ids: set
    ) -> List[CandidateChunk]:
        """Получает siblings по parent_chunk_id."""
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.paragraph_path,
                    c.parent_chunk_id,
                    c.text_raw,
                    c.page_start,
                    c.page_end
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.parent_chunk_id = %s
                  AND c.chunk_id != ALL(%s)
                ORDER BY c.anchor
            """, (doc_id, parent_chunk_id, list(seen_chunk_ids)))
            
            siblings = []
            for row in cur.fetchall():
                candidate = CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[7],
                    page_start=row[8],
                    page_end=row[9],
                    vector_score=0.0,
                    fts_score=0.0,
                    final_score=0.0
                )
                candidate.paragraph_path = row[5]
                candidate.parent_chunk_id = row[6]
                siblings.append(candidate)
            
            return siblings
    
    async def _get_siblings_by_section(
        self,
        section_id: str,
        doc_id: str,
        seen_chunk_ids: set
    ) -> List[CandidateChunk]:
        """Получает siblings по section_id (fallback)."""
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.paragraph_path,
                    c.parent_chunk_id,
                    c.text_raw,
                    c.page_start,
                    c.page_end
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.section_id = %s
                  AND c.chunk_id != ALL(%s)
                ORDER BY c.anchor
                LIMIT 20
            """, (doc_id, section_id, list(seen_chunk_ids)))
            
            siblings = []
            for row in cur.fetchall():
                candidate = CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[7],
                    page_start=row[8],
                    page_end=row[9],
                    vector_score=0.0,
                    fts_score=0.0,
                    final_score=0.0
                )
                candidate.paragraph_path = row[5]
                candidate.parent_chunk_id = row[6]
                siblings.append(candidate)
            
            return siblings
    
    def _dedup_and_select(
        self,
        candidates: List[CandidateChunk],
        k: int
    ) -> List[CandidateChunk]:
        """
        Дедупликация и выбор top-k с сортировкой.
        
        Сортировка:
        1. По section_id
        2. Внутри секции по anchor (чтобы пункты шли (a)(1)(2)(3))
        """
        # Дедуп по chunk_id
        seen = set()
        deduped = []
        for candidate in candidates:
            if candidate.chunk_id not in seen:
                deduped.append(candidate)
                seen.add(candidate.chunk_id)
        
        # Сортировка: сначала по section_id, затем по anchor
        deduped.sort(key=lambda x: (x.section_id, x.anchor or ""))
        
        # Ограничиваем k
        return deduped[:k]


class PenaltiesRetriever(BaseRetriever):
    """
    Ретривер для вопросов о санкциях/штрафах.
    
    Возвращает несколько релевантных подпунктов (обычно 3-8 atomic чанков) с суммами/диапазонами штрафов,
    категориями нарушений, условиями применения санкций.
    Использует гибридный поиск + усиление "денежности" + расширение по секции.
    """
    
    # Буст-слова для FTS query
    PENALTY_BOOST_WORDS = [
        "penalty", "penalties", "civil", "criminal", "fine", "fines",
        "violation", "violations", "noncompliance",
        "civil monetary penalty", "CMP",
        "amount", "amounts", "maximum", "minimum", "tier", "tiers",
        "annual", "calendar year",
        "dollar", "dollars",
        "45 CFR"
    ]
    
    def __init__(self, db_connection=None):
        """
        Инициализация ретривера штрафов.
        
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
        parts: Optional[List[int]] = None,
        k: int = 6,
        seed_k: int = 4,
        expand_section: bool = True,
        need_amounts: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Поиск информации о штрафах/санкциях.
        
        Args:
            question_embedding: Эмбеддинг вопроса (размерность 4096)
            max_results: Максимальное количество результатов (по умолчанию 6)
            question: Текст вопроса
            doc_id: ID документа (по умолчанию 'hipaa-reg-2013-03-26')
            parts: Список частей для поиска (по умолчанию [160, 164])
            k: Количество итоговых фрагментов (по умолчанию 6)
            seed_k: Количество seed-чанков (по умолчанию 4)
            expand_section: Добавлять соседние подпункты в той же секции (по умолчанию True)
            need_amounts: Усиливать наличие чисел/денежных сумм (по умолчанию True)
        
        Returns:
            Список словарей с информацией о штрафах
        """
        doc_id = doc_id or "hipaa-reg-2013-03-26"
        question = question or ""
        parts = parts or [160, 164]
        k = min(k, max_results, 10)  # Максимум 10 пунктов
        
        logger.info(f"PenaltiesRetriever: question='{question[:50]}...', k={k}, seed_k={seed_k}, expand={expand_section}, need_amounts={need_amounts}")
        
        # Step 0: Подготовка FTS query (усиление "penalty")
        fts_query = self._build_fts_query(question)
        logger.info(f"FTS query (усиленный): {fts_query[:100]}...")
        
        # Step 1: Candidate retrieval (hybrid) с жесткими фильтрами
        candidates = await self._get_candidates(
            question_embedding=question_embedding,
            fts_query=fts_query,
            doc_id=doc_id,
            parts=parts,
            limit=80
        )
        
        if not candidates:
            logger.warning("Не найдено кандидатов для штрафов")
            return []
        
        # Step 2: Amount scoring (усиление чанков с числами/суммами)
        for candidate in candidates:
            candidate.amount_score = self._calculate_amount_score(candidate.text_raw)
        
        # Применяем amount scoring к final_score
        for candidate in candidates:
            base_score = candidate.final_score
            if need_amounts:
                candidate.final_score = 0.55 * base_score + 0.45 * candidate.amount_score
            else:
                candidate.final_score = base_score
        
        # Step 3: Seed selection
        seeds = self._select_seeds(candidates, seed_k)
        logger.info(f"Выбрано seed-чанков: {len(seeds)}")
        
        # Step 4: Expansion (optional)
        all_results = list(seeds)
        
        if expand_section:
            expanded = await self._expand_by_section(seeds, doc_id, parts, need_amounts)
            logger.info(f"Найдено expanded-чанков: {len(expanded)}")
            all_results.extend(expanded)
        
        # Step 5: Итоговая сборка и сортировка
        final_results = self._dedup_and_select(all_results, k)
        
        logger.info(f"Финальных результатов: {len(final_results)}")
        
        # Конвертируем в формат API
        results = []
        for candidate in final_results:
            results.append({
                "chunk_id": candidate.chunk_id,
                "anchor": candidate.anchor,
                "section_id": candidate.section_id,
                "section_number": candidate.section_number,
                "section_title": candidate.section_title,
                "paragraph_path": getattr(candidate, 'paragraph_path', None),
                "text_raw": candidate.text_raw,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "scores": {
                    "vector_score": candidate.vector_score,
                    "fts_score": candidate.fts_score,
                    "amount_score": getattr(candidate, 'amount_score', 0.0),
                    "final_score": candidate.final_score
                },
                "explanation": getattr(candidate, 'explanation', 'penalty filter + hybrid retrieval')
            })
        
        return results
    
    def _build_fts_query(self, question: str) -> str:
        """
        Формирует усиленный FTS query из вопроса + буст-слова.
        """
        if not question:
            return " ".join(self.PENALTY_BOOST_WORDS)
        
        boost_text = " ".join(self.PENALTY_BOOST_WORDS)
        return f"{question} {boost_text}"
    
    async def _get_candidates(
        self,
        question_embedding: List[float],
        fts_query: str,
        doc_id: str,
        parts: List[int],
        limit: int
    ) -> List[CandidateChunk]:
        """
        Получает кандидатов через hybrid поиск (vector + FTS).
        """
        # Пробуем сначала с chunk_kind='penalty' (если есть в данных)
        # Если нет результатов, используем fallback на ('penalty','procedure','other')
        
        # 1B) Vector candidates
        vector_candidates = await self._vector_search_penalties(question_embedding, doc_id, parts, limit)
        logger.info(f"Vector кандидатов: {len(vector_candidates)}")
        
        # 1C) FTS candidates
        fts_candidates = await self._fts_search_penalties(fts_query, doc_id, parts, limit)
        logger.info(f"FTS кандидатов: {len(fts_candidates)}")
        
        # 1D) Merge + base scoring
        merged = self._merge_and_score_penalties(vector_candidates, fts_candidates)
        
        return merged
    
    async def _vector_search_penalties(
        self,
        question_embedding: List[float],
        doc_id: str,
        parts: List[int],
        limit: int
    ) -> List[CandidateChunk]:
        """Vector similarity поиск для штрафов."""
        with self.db.cursor() as cur:
            # Используем fallback на ('scope', 'requirement', 'other') так как 'penalty' нет в схеме
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.paragraph_path,
                    c.text_raw,
                    c.page_start,
                    c.page_end,
                    1 - (c.embedding <=> %s::vector) AS vector_score
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.part = ANY(%s)
                  AND c.embedding IS NOT NULL
                  AND c.chunk_kind IN ('scope', 'requirement', 'other')
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """
            
            cur.execute(query, (question_embedding, doc_id, parts, question_embedding, limit))
            
            candidates = []
            for row in cur.fetchall():
                candidate = CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[6],
                    page_start=row[7],
                    page_end=row[8],
                    vector_score=float(row[9]),
                    fts_score=0.0,
                    final_score=0.0
                )
                candidate.paragraph_path = row[5]
                candidates.append(candidate)
            
            return candidates
    
    async def _fts_search_penalties(
        self,
        fts_query: str,
        doc_id: str,
        parts: List[int],
        limit: int
    ) -> List[CandidateChunk]:
        """FTS поиск для штрафов."""
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
                        c.paragraph_path,
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
                      AND c.part = ANY(%s)
                      AND c.chunk_kind IN ('scope', 'requirement', 'other')
                      AND plainto_tsquery('english', %s) @@ to_tsvector('english', COALESCE(c.text_raw, ''))
                    ORDER BY fts_score DESC
                    LIMIT %s
                """
                
                cur.execute(query, (fts_query, doc_id, parts, fts_query, limit))
                
                candidates = []
                for row in cur.fetchall():
                    candidate = CandidateChunk(
                        chunk_id=row[0],
                        anchor=row[1],
                        section_id=row[2],
                        section_number=row[3],
                        section_title=row[4],
                        text_raw=row[6],
                        page_start=row[7],
                        page_end=row[8],
                        vector_score=0.0,
                        fts_score=float(row[9]) if row[9] else 0.0,
                        final_score=0.0
                    )
                    candidate.paragraph_path = row[5]
                    candidates.append(candidate)
                
                return candidates
            except Exception as e:
                logger.warning(f"FTS поиск штрафов не удался: {e}")
                self.db.rollback()
                return []
    
    def _merge_and_score_penalties(
        self,
        vector_candidates: List[CandidateChunk],
        fts_candidates: List[CandidateChunk]
    ) -> List[CandidateChunk]:
        """
        Объединяет vector и FTS кандидатов, нормализует и ранжирует.
        
        base_score = 0.65*vector_score + 0.35*fts_score_norm
        """
        # Нормализация FTS scores
        if fts_candidates:
            max_fts = max(c.fts_score for c in fts_candidates) if fts_candidates else 1.0
            if max_fts > 0:
                for c in fts_candidates:
                    c.fts_score = c.fts_score / max_fts
        
        # Создаем словарь по chunk_id для объединения
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
        
        # Вычисляем base_score для всех
        for candidate in merged_dict.values():
            candidate.final_score = 0.65 * candidate.vector_score + 0.35 * candidate.fts_score
        
        # Сортируем по base_score
        merged_list = sorted(merged_dict.values(), key=lambda x: x.final_score, reverse=True)
        
        return merged_list
    
    def _calculate_amount_score(self, text: str) -> float:
        """
        Вычисляет amount_score по regex на текст.
        
        Returns:
            1.0 если найден паттерн суммы ($, USD, числа с разделителями)
            0.5 если просто есть числа
            0.0 если нет чисел
        """
        import re
        
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
    
    def _select_seeds(self, candidates: List[CandidateChunk], seed_k: int) -> List[CandidateChunk]:
        """
        Выбирает seed_k лучших seeds с дедупом по section_id.
        """
        seen_sections = set()
        selected = []
        
        for candidate in candidates:
            if candidate.section_id not in seen_sections:
                selected.append(candidate)
                seen_sections.add(candidate.section_id)
                if len(selected) >= seed_k:
                    break
        
        return selected
    
    async def _expand_by_section(
        self,
        seeds: List[CandidateChunk],
        doc_id: str,
        parts: List[int],
        need_amounts: bool
    ) -> List[CandidateChunk]:
        """
        Расширяет seeds соседями по section_id.
        
        Берет все atomic чанки в той же section_id, или только те, где есть amount_score>0.
        """
        expanded = []
        seen_chunk_ids = {seed.chunk_id for seed in seeds}
        
        for seed in seeds:
            section_siblings = await self._get_section_siblings(
                seed.section_id, doc_id, parts, seen_chunk_ids, need_amounts
            )
            
            # Присваиваем оценку siblings (чуть ниже seed)
            for sibling in section_siblings:
                sibling.vector_score = seed.vector_score * 0.95
                sibling.fts_score = seed.fts_score * 0.95
                sibling.amount_score = self._calculate_amount_score(sibling.text_raw)
                sibling.final_score = seed.final_score * 0.95
                sibling.explanation = "expanded from same section"
                seen_chunk_ids.add(sibling.chunk_id)
            
            expanded.extend(section_siblings)
        
        return expanded
    
    async def _get_section_siblings(
        self,
        section_id: str,
        doc_id: str,
        parts: List[int],
        seen_chunk_ids: set,
        need_amounts: bool
    ) -> List[CandidateChunk]:
        """Получает siblings по section_id."""
        with self.db.cursor() as cur:
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.paragraph_path,
                    c.text_raw,
                    c.page_start,
                    c.page_end
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.part = ANY(%s)
                  AND c.section_id = %s
                  AND c.chunk_id != ALL(%s)
                ORDER BY c.anchor
                LIMIT 20
            """
            
            cur.execute(query, (doc_id, parts, section_id, list(seen_chunk_ids)))
            
            siblings = []
            for row in cur.fetchall():
                candidate = CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[6],
                    page_start=row[7],
                    page_end=row[8],
                    vector_score=0.0,
                    fts_score=0.0,
                    final_score=0.0
                )
                candidate.paragraph_path = row[5]
                
                # Если need_amounts=True, фильтруем только те, где есть amount_score>0
                if need_amounts:
                    amount_score = self._calculate_amount_score(candidate.text_raw)
                    if amount_score > 0:
                        candidate.amount_score = amount_score
                        siblings.append(candidate)
                else:
                    siblings.append(candidate)
            
            return siblings
    
    def _dedup_and_select(
        self,
        candidates: List[CandidateChunk],
        k: int
    ) -> List[CandidateChunk]:
        """
        Дедупликация и выбор top-k с сортировкой.
        
        Сортировка:
        1. По section_id
        2. Внутри секции по anchor
        """
        # Дедуп по chunk_id
        seen = set()
        deduped = []
        for candidate in candidates:
            if candidate.chunk_id not in seen:
                deduped.append(candidate)
                seen.add(candidate.chunk_id)
        
        # Сортировка: сначала по section_id, затем по anchor
        deduped.sort(key=lambda x: (x.section_id, x.anchor or ""))
        
        # Ограничиваем k
        return deduped[:k]


class ProceduralRetriever(BaseRetriever):
    """
    Ретривер для вопросов типа 'procedural / best practice'.
    
    Отвечает на вопросы вида "mentions/does it require/recommend best practices" для тем безопасности/процедур.
    Возвращает 2-6 релевантных atomic чанков с anchor и формирует сигнал yes/no для answer-слоя.
    """
    
    # Словари токенов для под-тем
    TOPIC_TOKENS = {
        "encryption": ["encrypt", "encryption", "decrypt", "cryptograph", "key management"],
        "security": ["security", "safeguard", "safeguards", "technical safeguard", "administrative safeguard", "physical safeguard"],
    }
    
    # Общие токены для требований/рекомендаций
    REQUIREMENT_TOKENS = [
        "required", "must", "addressable", "implementation specification",
        "standard", "reasonable and appropriate", "shall"
    ]
    
    # Обязательные security-слова для FTS
    SECURITY_BOOST_WORDS = [
        "security", "safeguard", "safeguards", "technical", "administrative",
        "physical", "implementation specification", "standard", "addressable"
    ]
    
    def __init__(self, db_connection=None):
        """
        Инициализация ретривера процедур/best practices.
        
        Args:
            db_connection: Подключение к PostgreSQL (если None, создается новое)
        """
        self.db = db_connection or get_db_connection()
        self.embedding_client = get_embedding_client()
    
    async def retrieve(
        self,
        question_embedding: List[float],
        max_results: int = 4,
        question: Optional[str] = None,
        doc_id: Optional[str] = None,
        parts: Optional[List[int]] = None,
        k: int = 4,
        seed_k: int = 3,
        expand_section: bool = True,
        need_yesno: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Поиск информации о процедурах/best practices.
        
        Args:
            question_embedding: Эмбеддинг вопроса (размерность 4096)
            max_results: Максимальное количество результатов (по умолчанию 4)
            question: Текст вопроса
            doc_id: ID документа (по умолчанию 'hipaa-reg-2013-03-26')
            parts: Список частей для поиска (по умолчанию [164])
            k: Количество итоговых пунктов (по умолчанию 4)
            seed_k: Количество seed-чанков (по умолчанию 3)
            expand_section: Добавлять соседние подпункты в той же секции (по умолчанию True)
            need_yesno: Формировать сигнал yes/no (по умолчанию True)
        
        Returns:
            Список словарей с информацией о процедурах + yesno_signal и yesno_rationale
        """
        doc_id = doc_id or "hipaa-reg-2013-03-26"
        question = question or ""
        parts = parts or [164]  # По умолчанию Part 164 (security/privacy)
        k = min(k, max_results, 6)  # Максимум 6 пунктов
        
        logger.info(f"ProceduralRetriever: question='{question[:50]}...', k={k}, seed_k={seed_k}, expand={expand_section}, need_yesno={need_yesno}")
        
        # Step 0: Определить под-тему и ключевые токены
        topic, topic_tokens = self._extract_topic_and_tokens(question)
        logger.info(f"Извлеченная тема: {topic}, токены: {topic_tokens[:5]}...")
        
        # Step 1: Подготовка FTS query (усиление)
        fts_query = self._build_fts_query(question, topic_tokens)
        logger.info(f"FTS query (усиленный): {fts_query[:100]}...")
        
        # Step 2: Candidate retrieval (hybrid) по atomic chunks
        candidates = await self._get_candidates(
            question_embedding=question_embedding,
            fts_query=fts_query,
            doc_id=doc_id,
            parts=parts,
            limit=80
        )
        
        # Fallback: если по Part 164 мало кандидатов (< 3), снять фильтр part
        if len(candidates) < 3 and parts == [164]:
            logger.warning(f"Мало кандидатов ({len(candidates)}), снимаем фильтр part")
            candidates = await self._get_candidates(
                question_embedding=question_embedding,
                fts_query=fts_query,
                doc_id=doc_id,
                parts=None,  # Без фильтра part
                limit=80
            )
        
        if not candidates:
            logger.warning("Не найдено кандидатов для процедур")
            return []
        
        # Step 3: Keyword evidence scoring
        for candidate in candidates:
            candidate.keyword_score = self._calculate_keyword_score(candidate.text_raw, topic_tokens)
        
        # Step 4: Merge + final scoring
        for candidate in candidates:
            # Нормализация уже сделана в _merge_and_score_procedural
            candidate.final_score = 0.55 * candidate.vector_score + 0.20 * candidate.fts_score + 0.25 * candidate.keyword_score
        
        # Сортируем по final_score
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # Step 5: Seed selection + expansion
        seeds = self._select_seeds(candidates, seed_k)
        logger.info(f"Выбрано seed-чанков: {len(seeds)}")
        
        all_results = list(seeds)
        
        if expand_section:
            expanded = await self._expand_by_section(seeds, doc_id, parts, topic_tokens)
            logger.info(f"Найдено expanded-чанков: {len(expanded)}")
            all_results.extend(expanded)
        
        # Дедупликация и выбор top-k
        final_results = self._dedup_and_select(all_results, k)
        
        logger.info(f"Финальных результатов: {len(final_results)}")
        
        # Step 6: Формирование сигнала yes/no
        yesno_signal = None
        yesno_rationale = None
        
        if need_yesno:
            yesno_signal, yesno_rationale = self._determine_yesno_signal(final_results, topic, topic_tokens)
            logger.info(f"Yes/No сигнал: {yesno_signal}, rationale: {yesno_rationale[:100]}...")
        
        # Конвертируем в формат API
        results = []
        for candidate in final_results:
            results.append({
                "chunk_id": candidate.chunk_id,
                "anchor": candidate.anchor,
                "section_id": candidate.section_id,
                "section_number": candidate.section_number,
                "section_title": candidate.section_title,
                "paragraph_path": getattr(candidate, 'paragraph_path', None),
                "text_raw": candidate.text_raw,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "scores": {
                    "vector_score": candidate.vector_score,
                    "fts_score": candidate.fts_score,
                    "keyword_score": getattr(candidate, 'keyword_score', 0.0),
                    "final_score": candidate.final_score
                },
                "explanation": getattr(candidate, 'explanation', 'procedural retrieval')
            })
        
        # Добавляем yesno_signal и yesno_rationale к первому результату или как отдельные поля
        if need_yesno and results:
            results[0]["yesno_signal"] = yesno_signal
            results[0]["yesno_rationale"] = yesno_rationale
        
        return results
    
    def _extract_topic_and_tokens(self, question: str) -> tuple[Optional[str], List[str]]:
        """
        Извлекает под-тему и ключевые токены из вопроса.
        
        Returns:
            (topic, topic_tokens) - например ("encryption", ["encrypt", "encryption", ...])
        """
        if not question:
            return None, []
        
        question_lower = question.lower()
        topic_tokens = []
        topic = None
        
        # Проверяем известные темы
        for topic_name, tokens in self.TOPIC_TOKENS.items():
            if any(token in question_lower for token in tokens):
                topic = topic_name
                topic_tokens.extend(tokens)
                break
        
        # Добавляем общие токены требований
        topic_tokens.extend(self.REQUIREMENT_TOKENS)
        
        # Добавляем общие security-токены
        topic_tokens.extend(self.SECURITY_BOOST_WORDS)
        
        return topic, list(set(topic_tokens))  # Дедупликация
    
    def _build_fts_query(self, question: str, topic_tokens: List[str]) -> str:
        """
        Формирует усиленный FTS query из вопроса + буст-слова + токены темы.
        """
        if not question:
            return " ".join(self.SECURITY_BOOST_WORDS + topic_tokens)
        
        boost_text = " ".join(self.SECURITY_BOOST_WORDS + topic_tokens)
        return f"{question} {boost_text}"
    
    async def _get_candidates(
        self,
        question_embedding: List[float],
        fts_query: str,
        doc_id: str,
        parts: Optional[List[int]],
        limit: int
    ) -> List[CandidateChunk]:
        """
        Получает кандидатов через hybrid поиск (vector + FTS).
        """
        # 2B) Vector candidates
        vector_candidates = await self._vector_search_procedural(question_embedding, doc_id, parts, limit)
        logger.info(f"Vector кандидатов: {len(vector_candidates)}")
        
        # 2C) FTS candidates
        fts_candidates = await self._fts_search_procedural(fts_query, doc_id, parts, limit)
        logger.info(f"FTS кандидатов: {len(fts_candidates)}")
        
        # Merge + base scoring
        merged = self._merge_and_score_procedural(vector_candidates, fts_candidates)
        
        return merged
    
    async def _vector_search_procedural(
        self,
        question_embedding: List[float],
        doc_id: str,
        parts: Optional[List[int]],
        limit: int
    ) -> List[CandidateChunk]:
        """Vector similarity поиск для процедур."""
        with self.db.cursor() as cur:
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.paragraph_path,
                    c.text_raw,
                    c.page_start,
                    c.page_end,
                    1 - (c.embedding <=> %s::vector) AS vector_score
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.embedding IS NOT NULL
            """
            params = [question_embedding, doc_id]
            
            if parts:
                query += " AND c.part = ANY(%s)"
                params.append(parts)
            
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
                    text_raw=row[6],
                    page_start=row[7],
                    page_end=row[8],
                    vector_score=float(row[9]),
                    fts_score=0.0,
                    final_score=0.0
                )
                candidate.paragraph_path = row[5]
                candidates.append(candidate)
            
            return candidates
    
    async def _fts_search_procedural(
        self,
        fts_query: str,
        doc_id: str,
        parts: Optional[List[int]],
        limit: int
    ) -> List[CandidateChunk]:
        """FTS поиск для процедур."""
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
                        c.paragraph_path,
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
                      AND plainto_tsquery('english', %s) @@ to_tsvector('english', COALESCE(c.text_raw, ''))
                """
                params = [fts_query, doc_id, fts_query]
                
                if parts:
                    query += " AND c.part = ANY(%s)"
                    params.append(parts)
                
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
                        text_raw=row[6],
                        page_start=row[7],
                        page_end=row[8],
                        vector_score=0.0,
                        fts_score=float(row[9]) if row[9] else 0.0,
                        final_score=0.0
                    )
                    candidate.paragraph_path = row[5]
                    candidates.append(candidate)
                
                return candidates
            except Exception as e:
                logger.warning(f"FTS поиск процедур не удался: {e}")
                self.db.rollback()
                return []
    
    def _merge_and_score_procedural(
        self,
        vector_candidates: List[CandidateChunk],
        fts_candidates: List[CandidateChunk]
    ) -> List[CandidateChunk]:
        """
        Объединяет vector и FTS кандидатов, нормализует.
        """
        # Нормализация FTS scores
        if fts_candidates:
            max_fts = max(c.fts_score for c in fts_candidates) if fts_candidates else 1.0
            if max_fts > 0:
                for c in fts_candidates:
                    c.fts_score = c.fts_score / max_fts
        
        # Создаем словарь по chunk_id для объединения
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
        
        return list(merged_dict.values())
    
    def _calculate_keyword_score(self, text: str, topic_tokens: List[str]) -> float:
        """
        Вычисляет keyword_score на основе наличия ключевых слов в text_raw.
        
        Returns:
            keyword_score в диапазоне 0..1
        """
        import re
        
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
    
    def _select_seeds(self, candidates: List[CandidateChunk], seed_k: int) -> List[CandidateChunk]:
        """
        Выбирает seed_k лучших seeds с дедупом по section_id.
        """
        seen_sections = set()
        selected = []
        
        for candidate in candidates:
            if candidate.section_id not in seen_sections:
                selected.append(candidate)
                seen_sections.add(candidate.section_id)
                if len(selected) >= seed_k:
                    break
        
        return selected
    
    async def _expand_by_section(
        self,
        seeds: List[CandidateChunk],
        doc_id: str,
        parts: Optional[List[int]],
        topic_tokens: List[str]
    ) -> List[CandidateChunk]:
        """
        Расширяет seeds соседями по section_id, но только если они содержат security/encryption tokens.
        """
        expanded = []
        seen_chunk_ids = {seed.chunk_id for seed in seeds}
        
        for seed in seeds:
            section_siblings = await self._get_section_siblings(
                seed.section_id, doc_id, parts, seen_chunk_ids, topic_tokens
            )
            
            # Присваиваем оценку siblings (чуть ниже seed)
            for sibling in section_siblings:
                sibling.vector_score = seed.vector_score * 0.95
                sibling.fts_score = seed.fts_score * 0.95
                sibling.keyword_score = self._calculate_keyword_score(sibling.text_raw, topic_tokens)
                sibling.final_score = 0.55 * sibling.vector_score + 0.20 * sibling.fts_score + 0.25 * sibling.keyword_score
                sibling.explanation = "expanded from same section"
                seen_chunk_ids.add(sibling.chunk_id)
            
            expanded.extend(section_siblings)
        
        return expanded
    
    async def _get_section_siblings(
        self,
        section_id: str,
        doc_id: str,
        parts: Optional[List[int]],
        seen_chunk_ids: set,
        topic_tokens: List[str]
    ) -> List[CandidateChunk]:
        """Получает siblings по section_id, фильтруя по keyword_score > 0."""
        with self.db.cursor() as cur:
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.paragraph_path,
                    c.text_raw,
                    c.page_start,
                    c.page_end
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.section_id = %s
                  AND c.chunk_id != ALL(%s::text[])
            """
            params = [doc_id, section_id, list(seen_chunk_ids)]
            
            if parts:
                query += " AND c.part = ANY(%s)"
                params.append(parts)
            
            query += " ORDER BY c.anchor LIMIT 5"
            
            cur.execute(query, params)
            
            siblings = []
            for row in cur.fetchall():
                candidate = CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[6],
                    page_start=row[7],
                    page_end=row[8],
                    vector_score=0.0,
                    fts_score=0.0,
                    final_score=0.0
                )
                candidate.paragraph_path = row[5]
                
                # Фильтруем только те, где есть keyword_score > 0
                keyword_score = self._calculate_keyword_score(candidate.text_raw, topic_tokens)
                if keyword_score > 0:
                    candidate.keyword_score = keyword_score
                    siblings.append(candidate)
            
            return siblings
    
    def _dedup_and_select(
        self,
        candidates: List[CandidateChunk],
        k: int
    ) -> List[CandidateChunk]:
        """
        Дедупликация и выбор top-k с сортировкой.
        """
        # Дедуп по chunk_id
        seen = set()
        deduped = []
        for candidate in candidates:
            if candidate.chunk_id not in seen:
                deduped.append(candidate)
                seen.add(candidate.chunk_id)
        
        # Сортировка: сначала по section_id, затем по anchor
        deduped.sort(key=lambda x: (x.section_id, x.anchor or ""))
        
        # Ограничиваем k
        return deduped[:k]
    
    def _determine_yesno_signal(
        self,
        results: List[CandidateChunk],
        topic: Optional[str],
        topic_tokens: List[str]
    ) -> tuple[str, str]:
        """
        Формирует сигнал yes/no на основе найденных чанков.
        
        Returns:
            (yesno_signal, yesno_rationale) - например ("yes", "Found explicit term 'encryption' in §164.312...")
        """
        if not results:
            return "unclear", "No relevant chunks found"
        
        # Проверяем top-k чанков
        text_combined = " ".join([r.text_raw.lower() for r in results[:3]])
        anchors = [r.anchor for r in results[:3] if r.anchor]
        
        # Проверка на явное "not required"
        import re
        if re.search(r'\b(not required|no requirement)\b', text_combined):
            anchor_str = anchors[0] if anchors else "relevant section"
            return "no", f"Found explicit 'not required' statement in {anchor_str}"
        
        # Проверка на прямое упоминание темы (например, encryption)
        # Проверяем все результаты, а не только combined text
        encryption_tokens = ["encrypt", "encryption", "decrypt", "cryptograph"]
        for result in results[:3]:
            text_lower = result.text_raw.lower()
            if any(token in text_lower for token in encryption_tokens):
                anchor_str = result.anchor if result.anchor else "relevant section"
                return "yes", f"Found explicit term 'encryption' in {anchor_str}"
        
        # Проверка на security/safeguards/implementation specification
        if re.search(r'\b(safeguard|safeguards|security|implementation specification)\b', text_combined):
            anchor_str = anchors[0] if anchors else "relevant section"
            return "unclear", f"Found safeguards/implementation specifications in {anchor_str}, but no explicit {topic or 'requirement'} mention"
        
        return "unclear", "Found relevant chunks but unclear evidence for explicit requirement"


class PermissionDisclosureRetriever(BaseRetriever):
    """
    Ретривер для вопросов типа 'permission / disclosure'.
    
    Для вопросов о допустимости раскрытия PHI возвращает набор атомарных параграфов из Part 164 (Privacy Rule),
    которые описывают разрешенные раскрытия, содержат условия, включают исключения/ограничения.
    """
    
    # Словари токенов для под-тем раскрытия
    DISCLOSURE_TOPICS = {
        "family": ["family", "relative", "relatives", "spouse", "friend", "involved", "care"],
        "law_enforcement": ["law enforcement", "police", "court", "subpoena", "warrant", "judicial"],
        "public_health": ["public health", "health department", "disease", "epidemic"],
        "employer": ["employer", "employment", "workplace"],
        "business_associate": ["business associate", "vendor", "contractor"],
        "research": ["research", "study", "researcher"],
    }
    
    # Общие disclosure-токены
    DISCLOSURE_TOKENS = [
        "disclose", "disclosure", "disclosures",
        "use", "uses",
        "permit", "permitted", "permissible",
        "authorization", "authorization required",
        "without authorization",
        "minimum necessary",
        "may", "must", "except", "subject to", "required by law"
    ]
    
    def __init__(self, db_connection=None):
        """
        Инициализация ретривера разрешений/раскрытий.
        
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
        part: int = 164,
        k: int = 6,
        seed_k: int = 4,
        expand_refs: bool = True,
        expand_siblings: bool = True,
        max_ref_hops: int = 1,
        ref_limit: int = 12,
        strict_quote: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Поиск информации о разрешениях/раскрытиях PHI.
        
        Args:
            question_embedding: Эмбеддинг вопроса (размерность 4096)
            max_results: Максимальное количество результатов (по умолчанию 6)
            question: Текст вопроса
            doc_id: ID документа (по умолчанию 'hipaa-reg-2013-03-26')
            part: Номер части (по умолчанию 164 - Privacy Rule)
            k: Количество итоговых результатов (по умолчанию 6)
            seed_k: Количество seed-чанков (по умолчанию 4)
            expand_refs: Расширять через references (по умолчанию True)
            expand_siblings: Расширять через siblings (по умолчанию True)
            max_ref_hops: Максимальная глубина reference expansion (по умолчанию 1)
            ref_limit: Количество ref-параграфов для подтягивания (по умолчанию 12)
            strict_quote: Требовать точную цитату (по умолчанию True)
        
        Returns:
            Список словарей с информацией о разрешениях/раскрытиях + policy_signal
        """
        doc_id = doc_id or "hipaa-reg-2013-03-26"
        question = question or ""
        k = min(k, max_results, 10)  # Максимум 10 пунктов
        
        logger.info(f"PermissionDisclosureRetriever: question='{question[:50]}...', k={k}, seed_k={seed_k}, part={part}")
        
        # Step 0: Topic extraction (кому/куда раскрытие)
        topic, topic_tokens = self._extract_disclosure_topic(question)
        logger.info(f"Извлеченная тема раскрытия: {topic}, токены: {topic_tokens[:5]}...")
        
        # Step 1: FTS query (disclosure-focused)
        fts_query = self._build_disclosure_fts_query(question, topic_tokens)
        logger.info(f"FTS query (усиленный): {fts_query[:100]}...")
        
        # Step 2: Seed retrieval (hybrid, atomic, Part 164)
        candidates = await self._get_seeds(
            question_embedding=question_embedding,
            fts_query=fts_query,
            doc_id=doc_id,
            part=part,
            limit=80
        )
        
        # Fallback: если по Part 164 мало кандидатов (< 3), снять фильтр part
        if len(candidates) < 3 and part == 164:
            logger.warning(f"Мало кандидатов ({len(candidates)}), снимаем фильтр part")
            candidates = await self._get_seeds(
                question_embedding=question_embedding,
                fts_query=fts_query,
                doc_id=doc_id,
                part=None,  # Без фильтра part
                limit=80
            )
            # Приоритизируем Part 164 в scoring
            for candidate in candidates:
                if hasattr(candidate, 'part') and candidate.part == 164:
                    candidate.final_score *= 1.1  # Буст для Part 164
        
        if not candidates:
            logger.warning("Не найдено кандидатов для разрешений/раскрытий")
            return []
        
        # Step 3: Clause / Evidence scoring
        for candidate in candidates:
            candidate.keyword_score = self._calculate_clause_score(candidate.text_raw, topic_tokens)
        
        # Step 4: Merge + final scoring
        for candidate in candidates:
            candidate.final_score = 0.50 * candidate.vector_score + 0.20 * candidate.fts_score + 0.30 * candidate.keyword_score
        
        # Сортируем по final_score
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # Выбираем seed_k лучших сидов
        seeds = self._select_seeds_disclosure(candidates, seed_k, topic_tokens)
        logger.info(f"Выбрано seed-чанков: {len(seeds)}")
        
        # Помечаем seeds
        for seed in seeds:
            seed.is_seed = True
            seed.is_ref = False
            seed.is_sibling = False
        
        all_results = list(seeds)
        
        # Step 5: Reference expansion через chunk_refs (если таблица есть)
        if expand_refs:
            refs = await self._expand_references(seeds, doc_id, ref_limit, max_ref_hops, topic_tokens)
            logger.info(f"Найдено ref-чанков: {len(refs)}")
            all_results.extend(refs)
        
        # Step 6: Sibling expansion
        if expand_siblings:
            siblings = await self._expand_siblings_disclosure(seeds, doc_id, part, topic_tokens)
            logger.info(f"Найдено sibling-чанков: {len(siblings)}")
            all_results.extend(siblings)
        
        # Step 7: Dedup + ordering + select top-k
        final_results = self._dedup_and_order(all_results, k)
        
        logger.info(f"Финальных результатов: {len(final_results)}")
        
        # Step 8: Policy signal (optional)
        policy_signal = self._determine_policy_signal(final_results)
        logger.info(f"Policy signal: {policy_signal}")
        
        # Конвертируем в формат API
        results = []
        for candidate in final_results:
            results.append({
                "chunk_id": candidate.chunk_id,
                "anchor": candidate.anchor,
                "section_id": candidate.section_id,
                "section_number": candidate.section_number,
                "section_title": candidate.section_title,
                "paragraph_path": getattr(candidate, 'paragraph_path', None),
                "parent_chunk_id": getattr(candidate, 'parent_chunk_id', None),
                "text_raw": candidate.text_raw,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "scores": {
                    "vector_score": candidate.vector_score,
                    "fts_score": candidate.fts_score,
                    "keyword_score": getattr(candidate, 'keyword_score', 0.0),
                    "final_score": candidate.final_score
                },
                "flags": {
                    "is_seed": getattr(candidate, 'is_seed', False),
                    "is_ref": getattr(candidate, 'is_ref', False),
                    "is_sibling": getattr(candidate, 'is_sibling', False)
                },
                "explanation": getattr(candidate, 'explanation', 'disclosure retrieval'),
                "policy_signal": policy_signal
            })
        
        return results
    
    def _extract_disclosure_topic(self, question: str) -> tuple[Optional[str], List[str]]:
        """
        Извлекает тему раскрытия из вопроса.
        
        Returns:
            (topic, topic_tokens) - например ("family", ["family", "relative", ...])
        """
        if not question:
            return None, []
        
        question_lower = question.lower()
        topic_tokens = []
        topic = None
        
        # Проверяем известные темы
        for topic_name, tokens in self.DISCLOSURE_TOPICS.items():
            if any(token in question_lower for token in tokens):
                topic = topic_name
                topic_tokens.extend(tokens)
                break
        
        # Добавляем общие disclosure-токены
        topic_tokens.extend(self.DISCLOSURE_TOKENS)
        
        return topic, list(set(topic_tokens))  # Дедупликация
    
    def _build_disclosure_fts_query(self, question: str, topic_tokens: List[str]) -> str:
        """
        Формирует усиленный FTS query из вопроса + disclosure_tokens + topic_tokens.
        """
        if not question:
            return " ".join(self.DISCLOSURE_TOKENS + topic_tokens)
        
        boost_text = " ".join(self.DISCLOSURE_TOKENS + topic_tokens)
        return f"{question} {boost_text}"
    
    async def _get_seeds(
        self,
        question_embedding: List[float],
        fts_query: str,
        doc_id: str,
        part: Optional[int],
        limit: int
    ) -> List[CandidateChunk]:
        """
        Получает seed-чанки через hybrid поиск (vector + FTS).
        """
        # 2A) Vector seeds
        vector_seeds = await self._vector_search_disclosure(question_embedding, doc_id, part, limit)
        logger.info(f"Vector seeds: {len(vector_seeds)}")
        
        # 2B) FTS seeds
        fts_seeds = await self._fts_search_disclosure(fts_query, doc_id, part, limit)
        logger.info(f"FTS seeds: {len(fts_seeds)}")
        
        # Merge + base scoring
        merged = self._merge_and_score_disclosure(vector_seeds, fts_seeds)
        
        return merged
    
    async def _vector_search_disclosure(
        self,
        question_embedding: List[float],
        doc_id: str,
        part: Optional[int],
        limit: int
    ) -> List[CandidateChunk]:
        """Vector similarity поиск для disclosure."""
        with self.db.cursor() as cur:
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.paragraph_path,
                    c.parent_chunk_id,
                    c.text_raw,
                    c.page_start,
                    c.page_end,
                    c.part,
                    1 - (c.embedding <=> %s::vector) AS vector_score
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.embedding IS NOT NULL
            """
            params = [question_embedding, doc_id]
            
            if part:
                query += " AND c.part = %s"
                params.append(part)
            
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
                    text_raw=row[7],
                    page_start=row[8],
                    page_end=row[9],
                    vector_score=float(row[11]),
                    fts_score=0.0,
                    final_score=0.0
                )
                candidate.paragraph_path = row[5]
                candidate.parent_chunk_id = row[6]
                candidate.part = row[10]
                candidates.append(candidate)
            
            return candidates
    
    async def _fts_search_disclosure(
        self,
        fts_query: str,
        doc_id: str,
        part: Optional[int],
        limit: int
    ) -> List[CandidateChunk]:
        """FTS поиск для disclosure."""
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
                        c.paragraph_path,
                        c.parent_chunk_id,
                        c.text_raw,
                        c.page_start,
                        c.page_end,
                        c.part,
                        ts_rank_cd(
                            to_tsvector('english', COALESCE(c.text_raw, '')),
                            plainto_tsquery('english', %s)
                        ) AS fts_score
                    FROM chunks c
                    WHERE c.doc_id = %s
                      AND c.granularity = 'atomic'
                      AND plainto_tsquery('english', %s) @@ to_tsvector('english', COALESCE(c.text_raw, ''))
                """
                params = [fts_query, doc_id, fts_query]
                
                if part:
                    query += " AND c.part = %s"
                    params.append(part)
                
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
                        text_raw=row[7],
                        page_start=row[8],
                        page_end=row[9],
                        vector_score=0.0,
                        fts_score=float(row[11]) if row[11] else 0.0,
                        final_score=0.0
                    )
                    candidate.paragraph_path = row[5]
                    candidate.parent_chunk_id = row[6]
                    candidate.part = row[10]
                    candidates.append(candidate)
                
                return candidates
            except Exception as e:
                logger.warning(f"FTS поиск disclosure не удался: {e}")
                self.db.rollback()
                return []
    
    def _merge_and_score_disclosure(
        self,
        vector_candidates: List[CandidateChunk],
        fts_candidates: List[CandidateChunk]
    ) -> List[CandidateChunk]:
        """
        Объединяет vector и FTS кандидатов, нормализует.
        """
        # Нормализация FTS scores
        if fts_candidates:
            max_fts = max(c.fts_score for c in fts_candidates) if fts_candidates else 1.0
            if max_fts > 0:
                for c in fts_candidates:
                    c.fts_score = c.fts_score / max_fts
        
        # Создаем словарь по chunk_id для объединения
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
        
        return list(merged_dict.values())
    
    def _calculate_clause_score(self, text: str, topic_tokens: List[str]) -> float:
        """
        Вычисляет clause/evidence score на основе паттернов в text_raw.
        
        Returns:
            keyword_score в диапазоне 0..1
        """
        import re
        
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
        
        # Запрещающее
        if re.search(r'\b(may not|prohibited)\b', text_lower):
            score += 0.3
        
        # Topic boost
        if topic_tokens:
            topic_matches = sum(1 for token in topic_tokens if token in text_lower)
            if topic_matches > 0:
                score += 0.6 * min(1.0, topic_matches / 3.0)  # Нормализуем по количеству совпадений
        
        # Нормализация в диапазон 0..1
        return max(0.0, min(1.0, score))
    
    def _select_seeds_disclosure(
        self,
        candidates: List[CandidateChunk],
        seed_k: int,
        topic_tokens: List[str]
    ) -> List[CandidateChunk]:
        """
        Выбирает seed_k лучших seeds с дедупом по section_id.
        
        Если вопрос узкий (есть topic tokens), разрешить 2 сида из одной секции,
        если anchors разные и оба содержат topic.
        """
        selected = []
        seen_sections = {}
        
        for candidate in candidates:
            section_id = candidate.section_id
            
            # Если вопрос узкий (есть topic tokens), разрешить 2 сида из одной секции
            if topic_tokens and section_id in seen_sections:
                if seen_sections[section_id] < 2:
                    # Проверяем, содержит ли кандидат topic tokens
                    text_lower = candidate.text_raw.lower()
                    if any(token in text_lower for token in topic_tokens):
                        # Проверяем, что anchor отличается
                        existing_anchors = [c.anchor for c in selected if c.section_id == section_id]
                        if candidate.anchor not in existing_anchors:
                            selected.append(candidate)
                            seen_sections[section_id] += 1
                            if len(selected) >= seed_k:
                                break
                continue
            
            # Обычный случай: один seed на секцию
            if section_id not in seen_sections:
                selected.append(candidate)
                seen_sections[section_id] = 1
                if len(selected) >= seed_k:
                    break
        
        return selected
    
    async def _expand_references(
        self,
        seeds: List[CandidateChunk],
        doc_id: str,
        ref_limit: int,
        max_ref_hops: int,
        topic_tokens: List[str]
    ) -> List[CandidateChunk]:
        """
        Расширяет seeds через chunk_refs (если таблица есть).
        """
        # Проверяем, существует ли таблица chunk_refs
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'chunk_refs'
                )
            """)
            table_exists = cur.fetchone()[0]
            
            if not table_exists:
                logger.debug("Таблица chunk_refs не существует, пропускаем reference expansion")
                return []
        
        refs = []
        seen_chunk_ids = {seed.chunk_id for seed in seeds}
        
        for seed in seeds:
            # 5A) Получаем ref chunk_ids
            with self.db.cursor() as cur:
                cur.execute("""
                    SELECT r.to_chunk_id
                    FROM chunk_refs r
                    WHERE r.from_chunk_id = %s
                      AND r.to_chunk_id IS NOT NULL
                    LIMIT %s
                """, (seed.chunk_id, ref_limit))
                
                ref_chunk_ids = [row[0] for row in cur.fetchall()]
                
                if not ref_chunk_ids:
                    continue
                
                # 5B) Подтягиваем ref chunks
                cur.execute("""
                    SELECT 
                        c.chunk_id,
                        c.anchor,
                        c.section_id,
                        c.section_number,
                        c.section_title,
                        c.paragraph_path,
                        c.parent_chunk_id,
                        c.text_raw,
                        c.page_start,
                        c.page_end
                    FROM chunks c
                    WHERE c.chunk_id = ANY(%s::text[])
                """, (ref_chunk_ids,))
                
                for row in cur.fetchall():
                    if row[0] in seen_chunk_ids:
                        continue
                    
                    candidate = CandidateChunk(
                        chunk_id=row[0],
                        anchor=row[1],
                        section_id=row[2],
                        section_number=row[3],
                        section_title=row[4],
                        text_raw=row[7],
                        page_start=row[8],
                        page_end=row[9],
                        vector_score=seed.vector_score * 0.93,
                        fts_score=seed.fts_score * 0.93,
                        final_score=seed.final_score * 0.93
                    )
                    candidate.paragraph_path = row[5]
                    candidate.parent_chunk_id = row[6]
                    candidate.is_seed = False
                    candidate.is_ref = True
                    candidate.is_sibling = False
                    candidate.keyword_score = self._calculate_clause_score(candidate.text_raw, topic_tokens)
                    candidate.final_score = 0.50 * candidate.vector_score + 0.20 * candidate.fts_score + 0.30 * candidate.keyword_score
                    candidate.explanation = "reference expansion"
                    seen_chunk_ids.add(candidate.chunk_id)
                    refs.append(candidate)
        
        return refs
    
    async def _expand_siblings_disclosure(
        self,
        seeds: List[CandidateChunk],
        doc_id: str,
        part: Optional[int],
        topic_tokens: List[str]
    ) -> List[CandidateChunk]:
        """
        Расширяет seeds соседями (siblings) по parent_chunk_id или section_id.
        """
        siblings = []
        seen_chunk_ids = {seed.chunk_id for seed in seeds}
        
        for seed in seeds:
            seed_siblings = []
            
            # Если есть parent_chunk_id - подтянуть siblings по parent
            if hasattr(seed, 'parent_chunk_id') and seed.parent_chunk_id:
                seed_siblings = await self._get_siblings_by_parent_disclosure(
                    seed.parent_chunk_id, doc_id, part, seen_chunk_ids
                )
            
            # Иначе - подтянуть 1-3 соседних atomic в той же section_id
            if not seed_siblings:
                seed_siblings = await self._get_siblings_by_section_disclosure(
                    seed.section_id, doc_id, part, seen_chunk_ids, limit=3
                )
            
            # Присваиваем оценку siblings
            for sibling in seed_siblings:
                sibling.vector_score = seed.vector_score * 0.95
                sibling.fts_score = seed.fts_score * 0.95
                sibling.keyword_score = self._calculate_clause_score(sibling.text_raw, topic_tokens)
                sibling.final_score = 0.50 * sibling.vector_score + 0.20 * sibling.fts_score + 0.30 * sibling.keyword_score
                sibling.is_seed = False
                sibling.is_ref = False
                sibling.is_sibling = True
                sibling.explanation = "sibling expansion"
                seen_chunk_ids.add(sibling.chunk_id)
            
            siblings.extend(seed_siblings)
        
        return siblings
    
    async def _get_siblings_by_parent_disclosure(
        self,
        parent_chunk_id: str,
        doc_id: str,
        part: Optional[int],
        seen_chunk_ids: set
    ) -> List[CandidateChunk]:
        """Получает siblings по parent_chunk_id."""
        with self.db.cursor() as cur:
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.paragraph_path,
                    c.parent_chunk_id,
                    c.text_raw,
                    c.page_start,
                    c.page_end
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.parent_chunk_id = %s
                  AND c.chunk_id != ALL(%s::text[])
            """
            params = [doc_id, parent_chunk_id, list(seen_chunk_ids)]
            
            if part:
                query += " AND c.part = %s"
                params.append(part)
            
            query += " ORDER BY c.anchor"
            
            cur.execute(query, params)
            
            siblings = []
            for row in cur.fetchall():
                candidate = CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[7],
                    page_start=row[8],
                    page_end=row[9],
                    vector_score=0.0,
                    fts_score=0.0,
                    final_score=0.0
                )
                candidate.paragraph_path = row[5]
                candidate.parent_chunk_id = row[6]
                siblings.append(candidate)
            
            return siblings
    
    async def _get_siblings_by_section_disclosure(
        self,
        section_id: str,
        doc_id: str,
        part: Optional[int],
        seen_chunk_ids: set,
        limit: int = 3
    ) -> List[CandidateChunk]:
        """Получает siblings по section_id (ограниченно)."""
        with self.db.cursor() as cur:
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.paragraph_path,
                    c.parent_chunk_id,
                    c.text_raw,
                    c.page_start,
                    c.page_end
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.section_id = %s
                  AND c.chunk_id != ALL(%s::text[])
            """
            params = [doc_id, section_id, list(seen_chunk_ids)]
            
            if part:
                query += " AND c.part = %s"
                params.append(part)
            
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
                    text_raw=row[7],
                    page_start=row[8],
                    page_end=row[9],
                    vector_score=0.0,
                    fts_score=0.0,
                    final_score=0.0
                )
                candidate.paragraph_path = row[5]
                candidate.parent_chunk_id = row[6]
                siblings.append(candidate)
            
            return siblings
    
    def _dedup_and_order(
        self,
        candidates: List[CandidateChunk],
        k: int
    ) -> List[CandidateChunk]:
        """
        Дедупликация и сортировка.
        
        Сортировка:
        1. По section_id
        2. Внутри секции по anchor
        """
        # Дедуп по chunk_id
        seen = set()
        deduped = []
        for candidate in candidates:
            if candidate.chunk_id not in seen:
                deduped.append(candidate)
                seen.add(candidate.chunk_id)
        
        # Сортировка: сначала по section_id, затем по anchor
        deduped.sort(key=lambda x: (x.section_id, x.anchor or ""))
        
        # Ограничиваем k
        return deduped[:k]
    
    def _determine_policy_signal(self, results: List[CandidateChunk]) -> str:
        """
        Определяет policy_signal на основе найденных чанков.
        
        Returns:
            "permitted" | "prohibited" | "conditional" | "unclear"
        """
        if not results:
            return "unclear"
        
        import re
        
        # Проверяем top-k чанков
        text_combined = " ".join([r.text_raw.lower() for r in results[:3]])
        
        # Проверка на запрет
        if re.search(r'\b(may not|prohibited|not permitted)\b', text_combined):
            return "prohibited"
        
        # Проверка на разрешение с условиями
        has_may_disclose = re.search(r'\b(may disclose|may use or disclose|is permitted)\b', text_combined)
        has_conditions = re.search(r'\b(except|subject to|provided that|only if)\b', text_combined)
        
        if has_may_disclose and has_conditions:
            return "conditional"
        
        if has_may_disclose:
            return "permitted"
        
        return "unclear"


class GeneralRetriever(BaseRetriever):
    """
    Ретривер для прочих вопросов (General Fallback).
    
    Обрабатывает любые "прочие" вопросы, которые не являются определением/навигацией/штрафами/
    дисклозурами/строгими цитатами. Возвращает достаточный контекст с цитируемостью (anchors),
    используя гибридный поиск (vector + FTS) и иерархическую структуру (atomic + родительская секция).
    """
    
    def __init__(self, db_connection=None):
        """
        Инициализация общего ретривера.
        
        Args:
            db_connection: Подключение к PostgreSQL (если None, создается новое)
        """
        self.db = db_connection or get_db_connection()
        self.embedding_client = get_embedding_client()
    
    async def retrieve(
        self,
        question_embedding: List[float],
        max_results: int = 8,
        question: Optional[str] = None,
        doc_id: Optional[str] = None,
        k: int = 8,
        seed_k: int = 6,
        fts_weight: float = 0.35,
        vector_weight: float = 0.65,
        include_parent_sections: bool = True,
        parent_limit: int = 2,
        max_per_section: int = 2,
        use_part_hint: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Поиск для прочих вопросов (general fallback).
        
        Args:
            question_embedding: Эмбеддинг вопроса (размерность 4096)
            max_results: Максимальное количество результатов (по умолчанию 8)
            question: Текст вопроса
            doc_id: ID документа (по умолчанию 'hipaa-reg-2013-03-26')
            k: Количество итоговых результатов (по умолчанию 8)
            seed_k: Количество первичных кандидатов (по умолчанию 6)
            fts_weight: Вес FTS score (по умолчанию 0.35)
            vector_weight: Вес vector score (по умолчанию 0.65)
            include_parent_sections: Добавлять parent/section chunks (по умолчанию True)
            parent_limit: Максимум уникальных parent chunks (по умолчанию 2)
            max_per_section: Максимум atomic чанков из одной section_id (по умолчанию 2)
            use_part_hint: Использовать part hint (по умолчанию True)
        
        Returns:
            Список словарей с информацией о чанках + coverage_note
        """
        doc_id = doc_id or "hipaa-reg-2013-03-26"
        question = question or ""
        k = min(k, max_results, 12)  # Максимум 12 результатов
        
        logger.info(f"GeneralRetriever: question='{question[:50]}...', k={k}, seed_k={seed_k}, max_per_section={max_per_section}")
        
        # Step 0: Part hint (легкое предположение)
        part_hint = None
        if use_part_hint:
            part_hint = self._determine_part_hint(question)
            logger.info(f"Part hint: {part_hint}")
        
        # Step 1: Candidate retrieval (hybrid) по atomic
        candidates = await self._get_candidates_general(
            question_embedding=question_embedding,
            question=question,
            doc_id=doc_id,
            part_hint=part_hint,
            limit=120
        )
        
        # Fallback: если с part_hint мало кандидатов, пробуем без него
        if len(candidates) < 3 and part_hint:
            logger.warning(f"Мало кандидатов с part_hint={part_hint}, пробуем без него")
            candidates = await self._get_candidates_general(
                question_embedding=question_embedding,
                question=question,
                doc_id=doc_id,
                part_hint=None,
                limit=120
            )
        
        # Fallback 2: если вообще ничего не нашлось, пробуем section chunks
        if not candidates:
            logger.warning("Не найдено atomic кандидатов, пробуем section chunks")
            candidates = await self._get_section_candidates_fallback(
                question_embedding=question_embedding,
                question=question,
                doc_id=doc_id,
                limit=20
            )
        
        if not candidates:
            logger.warning("Не найдено кандидатов для general retrieval")
            return []
        
        # Step 2: Merge + scoring
        merged = self._merge_and_score_general(candidates, vector_weight, fts_weight)
        
        # Step 3: Diversity constraint (важно!)
        seeds = self._select_diverse_seeds(merged, seed_k, max_per_section)
        logger.info(f"Выбрано diverse seed-чанков: {len(seeds)}")
        
        # Помечаем seeds
        for seed in seeds:
            seed.is_seed = True
            seed.is_parent = False
            seed.is_context = False
        
        all_results = list(seeds)
        
        # Step 4: Context enrichment: добавить parent/section chunks
        if include_parent_sections:
            parents = await self._get_parent_sections(seeds, doc_id, parent_limit)
            logger.info(f"Найдено parent-чанков: {len(parents)}")
            all_results.extend(parents)
        
        # Step 5: Final assembly
        final_results = self._final_assembly(all_results, k)
        
        logger.info(f"Финальных результатов: {len(final_results)}")
        
        # Подсчитываем coverage
        unique_sections = len(set(r.section_id for r in final_results))
        coverage_note = f"Top contexts span {unique_sections} sections; use anchors for citations."
        
        # Конвертируем в формат API
        results = []
        for candidate in final_results:
            results.append({
                "chunk_id": candidate.chunk_id,
                "anchor": candidate.anchor,
                "section_id": candidate.section_id,
                "section_number": candidate.section_number,
                "section_title": candidate.section_title,
                "granularity": getattr(candidate, 'granularity', 'atomic'),
                "paragraph_path": getattr(candidate, 'paragraph_path', None),
                "parent_chunk_id": getattr(candidate, 'parent_chunk_id', None),
                "text_raw": candidate.text_raw,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "scores": {
                    "vector_score": candidate.vector_score,
                    "fts_score": candidate.fts_score,
                    "final_score": candidate.final_score
                },
                "flags": {
                    "is_seed": getattr(candidate, 'is_seed', False),
                    "is_parent": getattr(candidate, 'is_parent', False),
                    "is_context": getattr(candidate, 'is_context', False)
                },
                "explanation": getattr(candidate, 'explanation', 'general retrieval'),
                "coverage_note": coverage_note
            })
        
        return results
    
    def _determine_part_hint(self, question: str) -> Optional[int]:
        """
        Определяет part hint из вопроса простым правилом.
        
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
    
    async def _get_candidates_general(
        self,
        question_embedding: List[float],
        question: str,
        doc_id: str,
        part_hint: Optional[int],
        limit: int
    ) -> List[CandidateChunk]:
        """
        Получает кандидатов через hybrid поиск (vector + FTS) по atomic.
        """
        # 1A) Vector candidates
        vector_candidates = await self._vector_search_general(question_embedding, doc_id, part_hint, limit)
        logger.info(f"Vector кандидатов: {len(vector_candidates)}")
        
        # 1B) FTS candidates
        fts_candidates = await self._fts_search_general(question, doc_id, part_hint, limit)
        logger.info(f"FTS кандидатов: {len(fts_candidates)}")
        
        # Объединяем
        all_candidates = vector_candidates + fts_candidates
        
        return all_candidates
    
    async def _vector_search_general(
        self,
        question_embedding: List[float],
        doc_id: str,
        part_hint: Optional[int],
        limit: int
    ) -> List[CandidateChunk]:
        """Vector search по atomic."""
        with self.db.cursor() as cur:
            query = """
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.granularity,
                    c.paragraph_path,
                    c.parent_chunk_id,
                    c.text_raw,
                    c.page_start,
                    c.page_end,
                    1 - (c.embedding <=> %s::vector) AS vector_score
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'atomic'
                  AND c.embedding IS NOT NULL
            """
            params = [question_embedding, doc_id]
            
            if part_hint:
                query += " AND c.part = %s"
                params.append(part_hint)
            
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
                    text_raw=row[8],
                    page_start=row[9],
                    page_end=row[10],
                    vector_score=float(row[11]),
                    fts_score=0.0,
                    final_score=0.0
                )
                candidate.granularity = row[5]
                candidate.paragraph_path = row[6]
                candidate.parent_chunk_id = row[7]
                candidates.append(candidate)
            
            return candidates
    
    async def _fts_search_general(
        self,
        question: str,
        doc_id: str,
        part_hint: Optional[int],
        limit: int
    ) -> List[CandidateChunk]:
        """FTS search по atomic."""
        with self.db.cursor() as cur:
            if not question or not question.strip():
                return []
            
            try:
                query = """
                    SELECT 
                        c.chunk_id,
                        c.anchor,
                        c.section_id,
                        c.section_number,
                        c.section_title,
                        c.granularity,
                        c.paragraph_path,
                        c.parent_chunk_id,
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
                      AND plainto_tsquery('english', %s) @@ to_tsvector('english', COALESCE(c.text_raw, ''))
                """
                params = [question, doc_id, question]
                
                if part_hint:
                    query += " AND c.part = %s"
                    params.append(part_hint)
                
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
                        text_raw=row[8],
                        page_start=row[9],
                        page_end=row[10],
                        vector_score=0.0,
                        fts_score=float(row[11]) if row[11] else 0.0,
                        final_score=0.0
                    )
                    candidate.granularity = row[5]
                    candidate.paragraph_path = row[6]
                    candidate.parent_chunk_id = row[7]
                    candidates.append(candidate)
                
                return candidates
            except Exception as e:
                logger.warning(f"FTS поиск general не удался: {e}")
                self.db.rollback()
                return []
    
    async def _get_section_candidates_fallback(
        self,
        question_embedding: List[float],
        question: str,
        doc_id: str,
        limit: int
    ) -> List[CandidateChunk]:
        """Fallback: поиск по section chunks если atomic не дал результатов."""
        with self.db.cursor() as cur:
            # Vector search по section chunks
            cur.execute("""
                SELECT 
                    c.chunk_id,
                    c.anchor,
                    c.section_id,
                    c.section_number,
                    c.section_title,
                    c.granularity,
                    c.paragraph_path,
                    c.parent_chunk_id,
                    c.text_raw,
                    c.page_start,
                    c.page_end,
                    1 - (c.embedding <=> %s::vector) AS vector_score
                FROM chunks c
                WHERE c.doc_id = %s
                  AND c.granularity = 'section'
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """, (question_embedding, doc_id, question_embedding, limit))
            
            candidates = []
            for row in cur.fetchall():
                candidate = CandidateChunk(
                    chunk_id=row[0],
                    anchor=row[1],
                    section_id=row[2],
                    section_number=row[3],
                    section_title=row[4],
                    text_raw=row[8],
                    page_start=row[9],
                    page_end=row[10],
                    vector_score=float(row[11]),
                    fts_score=0.0,
                    final_score=0.0
                )
                candidate.granularity = row[5]
                candidate.paragraph_path = row[6]
                candidate.parent_chunk_id = row[7]
                candidate.is_context = True
                candidates.append(candidate)
            
            return candidates
    
    def _merge_and_score_general(
        self,
        candidates: List[CandidateChunk],
        vector_weight: float,
        fts_weight: float
    ) -> List[CandidateChunk]:
        """
        Объединяет кандидатов, нормализует и ранжирует.
        
        final_score = vector_weight*vector_score + fts_weight*fts_score_norm
        """
        # Нормализация FTS scores
        fts_candidates = [c for c in candidates if c.fts_score > 0]
        if fts_candidates:
            max_fts = max(c.fts_score for c in fts_candidates)
            if max_fts > 0:
                for c in fts_candidates:
                    c.fts_score = c.fts_score / max_fts
        
        # Создаем словарь по chunk_id для объединения
        merged_dict: Dict[str, CandidateChunk] = {}
        
        for candidate in candidates:
            if candidate.chunk_id in merged_dict:
                # Объединяем оценки
                existing = merged_dict[candidate.chunk_id]
                if candidate.vector_score > 0:
                    existing.vector_score = candidate.vector_score
                if candidate.fts_score > 0:
                    existing.fts_score = candidate.fts_score
            else:
                merged_dict[candidate.chunk_id] = candidate
        
        # Вычисляем final_score для всех
        for candidate in merged_dict.values():
            candidate.final_score = vector_weight * candidate.vector_score + fts_weight * candidate.fts_score
        
        # Сортируем по final_score
        merged_list = sorted(merged_dict.values(), key=lambda x: x.final_score, reverse=True)
        
        return merged_list
    
    def _select_diverse_seeds(
        self,
        candidates: List[CandidateChunk],
        seed_k: int,
        max_per_section: int
    ) -> List[CandidateChunk]:
        """
        Выбирает diverse seeds с ограничением max_per_section из одной section_id.
        """
        selected = []
        section_counts = {}
        
        for candidate in candidates:
            section_id = candidate.section_id
            
            # Проверяем лимит по секции
            if section_id in section_counts:
                if section_counts[section_id] >= max_per_section:
                    continue
            
            selected.append(candidate)
            section_counts[section_id] = section_counts.get(section_id, 0) + 1
            
            if len(selected) >= seed_k:
                break
        
        return selected
    
    async def _get_parent_sections(
        self,
        seeds: List[CandidateChunk],
        doc_id: str,
        parent_limit: int
    ) -> List[CandidateChunk]:
        """
        Получает parent/section chunks для enrichment контекста.
        
        Для каждого atomic chunk:
        - если parent_chunk_id есть - подтянуть родителя (granularity='section')
        - иначе - подтянуть section-level chunk той же section_id
        """
        parent_ids = set()
        section_ids = set()
        
        for seed in seeds:
            if hasattr(seed, 'parent_chunk_id') and seed.parent_chunk_id:
                parent_ids.add(seed.parent_chunk_id)
            else:
                section_ids.add(seed.section_id)
        
        parents = []
        seen_parent_ids = set()
        
        # Подтягиваем parent chunks по parent_chunk_id
        if parent_ids:
            with self.db.cursor() as cur:
                cur.execute("""
                    SELECT 
                        c.chunk_id,
                        c.anchor,
                        c.section_id,
                        c.section_number,
                        c.section_title,
                        c.granularity,
                        c.paragraph_path,
                        c.parent_chunk_id,
                        c.text_raw,
                        c.page_start,
                        c.page_end
                    FROM chunks c
                    WHERE c.chunk_id = ANY(%s::text[])
                      AND c.granularity = 'section'
                """, (list(parent_ids),))
                
                for row in cur.fetchall():
                    if row[0] in seen_parent_ids:
                        continue
                    
                    candidate = CandidateChunk(
                        chunk_id=row[0],
                        anchor=row[1],
                        section_id=row[2],
                        section_number=row[3],
                        section_title=row[4],
                        text_raw=row[8],
                        page_start=row[9],
                        page_end=row[10],
                        vector_score=0.0,
                        fts_score=0.0,
                        final_score=0.0
                    )
                    candidate.granularity = row[5]
                    candidate.paragraph_path = row[6]
                    candidate.parent_chunk_id = row[7]
                    candidate.is_seed = False
                    candidate.is_parent = True
                    candidate.is_context = False
                    # final_score = max(child_scores)*0.9
                    child_scores = [s.final_score for s in seeds if hasattr(s, 'parent_chunk_id') and s.parent_chunk_id == row[0]]
                    if child_scores:
                        candidate.final_score = max(child_scores) * 0.9
                    candidate.explanation = "parent section context"
                    seen_parent_ids.add(candidate.chunk_id)
                    parents.append(candidate)
                    
                    if len(parents) >= parent_limit:
                        break
        
        # Подтягиваем section-level chunks по section_id (если parent_chunk_id не было)
        if len(parents) < parent_limit and section_ids:
            with self.db.cursor() as cur:
                cur.execute("""
                    SELECT 
                        c.chunk_id,
                        c.anchor,
                        c.section_id,
                        c.section_number,
                        c.section_title,
                        c.granularity,
                        c.paragraph_path,
                        c.parent_chunk_id,
                        c.text_raw,
                        c.page_start,
                        c.page_end
                    FROM chunks c
                    WHERE c.section_id = ANY(%s::text[])
                      AND c.granularity = 'section'
                      AND c.chunk_id != ALL(%s::text[])
                    LIMIT %s
                """, (list(section_ids), list(seen_parent_ids), parent_limit - len(parents)))
                
                for row in cur.fetchall():
                    candidate = CandidateChunk(
                        chunk_id=row[0],
                        anchor=row[1],
                        section_id=row[2],
                        section_number=row[3],
                        section_title=row[4],
                        text_raw=row[8],
                        page_start=row[9],
                        page_end=row[10],
                        vector_score=0.0,
                        fts_score=0.0,
                        final_score=0.0
                    )
                    candidate.granularity = row[5]
                    candidate.paragraph_path = row[6]
                    candidate.parent_chunk_id = row[7]
                    candidate.is_seed = False
                    candidate.is_parent = True
                    candidate.is_context = False
                    # final_score = max(child_scores)*0.9
                    child_scores = [s.final_score for s in seeds if s.section_id == row[2]]
                    if child_scores:
                        candidate.final_score = max(child_scores) * 0.9
                    candidate.explanation = "section context"
                    parents.append(candidate)
                    
                    if len(parents) >= parent_limit:
                        break
        
        return parents
    
    def _final_assembly(
        self,
        candidates: List[CandidateChunk],
        k: int
    ) -> List[CandidateChunk]:
        """
        Финальная сборка с сортировкой.
        
        Сортировка:
        1. По section_id
        2. Внутри section_id сначала section chunk (если есть), потом atomic по anchor
        """
        # Дедуп по chunk_id
        seen = set()
        deduped = []
        for candidate in candidates:
            if candidate.chunk_id not in seen:
                deduped.append(candidate)
                seen.add(candidate.chunk_id)
        
        # Сортировка: сначала по section_id, затем по granularity (section перед atomic), затем по anchor
        def sort_key(c):
            granularity_order = 0 if getattr(c, 'granularity', 'atomic') == 'section' else 1
            return (c.section_id, granularity_order, c.anchor or "")
        
        deduped.sort(key=sort_key)
        
        # Ограничиваем k
        return deduped[:k]


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
        
        # Step 0: Зафиксировать "юрисдикцию" (scope)
        if not anchor_prefix:
            anchor_prefix = self._determine_anchor_prefix(question)
        
        anchor_like = f"{anchor_prefix}%" if anchor_prefix else None
        
        logger.info(f"CitationRetriever: question='{question[:50]}...', anchor_prefix={anchor_prefix}, k={k}")
        
        # Step 1: Candidate retrieval: vector search по atomic + hard filter по anchor prefix
        vector_candidates = await self._vector_search_citation(
            question_embedding=question_embedding,
            doc_id=doc_id,
            anchor_like=anchor_like,
            limit=50
        )
        logger.info(f"Vector кандидатов: {len(vector_candidates)}")
        
        # Step 2 (optional): FTS внутри уже отфильтрованного anchor scope
        fts_candidates = []
        if question and question.strip():
            fts_query = self._build_citation_fts_query(question)
            fts_candidates = await self._fts_search_citation(
                fts_query=fts_query,
                doc_id=doc_id,
                anchor_like=anchor_like,
                limit=50
            )
            logger.info(f"FTS кандидатов: {len(fts_candidates)}")
        
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
        if expand_section and anchor_like:
            expanded = await self._expand_coverage(seeds, doc_id, anchor_like, k)
            logger.info(f"Найдено expanded-чанков: {len(expanded)}")
            all_results.extend(expanded)
        
        # Дедупликация и выбор top-k
        final_results = self._dedup_and_select_citation(all_results, k)
        
        logger.info(f"Финальных результатов: {len(final_results)}")
        
        # Конвертируем в формат API (строго anchor + text_raw)
        results = []
        for candidate in final_results:
            results.append({
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
            })
        
        return results
    
    def _determine_anchor_prefix(self, question: str) -> str:
        """
        Определяет anchor prefix по вопросу.
        
        Returns:
            anchor prefix (например, "§164.512")
        """
        if not question:
            return "§164.512"  # Дефолт для law enforcement
        
        question_lower = question.lower()
        
        # Проверяем известные темы
        for topic, prefix in self.TOPIC_ANCHOR_PREFIXES.items():
            if topic in question_lower:
                return prefix
        
        # Дефолт для law enforcement (наиболее частый кейс)
        return "§164.512"
    
    def _build_citation_fts_query(self, question: str) -> str:
        """
        Формирует FTS query с усилением для citation.
        """
        boost_words = [
            "law enforcement", "police", "court", "warrant",
            "subpoena", "administrative request", "disclosure", "disclose"
        ]
        
        boost_text = " ".join(boost_words)
        return f"{question} {boost_text}"
    
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
        # Нормализация FTS scores
        if fts_candidates:
            max_fts = max(c.fts_score for c in fts_candidates) if fts_candidates else 1.0
            if max_fts > 0:
                for c in fts_candidates:
                    c.fts_score = c.fts_score / max_fts
        
        # Создаем словарь по chunk_id для объединения
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
        
        # Вычисляем final_score для всех
        for candidate in merged_dict.values():
            if candidate.fts_score > 0:
                candidate.final_score = 0.8 * candidate.vector_score + 0.2 * candidate.fts_score
            else:
                candidate.final_score = candidate.vector_score  # Только vector если FTS нет
        
        # Сортируем по final_score
        merged_list = sorted(merged_dict.values(), key=lambda x: x.final_score, reverse=True)
        
        return merged_list
    
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
            
            # Берем ближайших соседей к seeds (по порядку anchor)
            expanded = []
            seed_anchors = [s.anchor for s in seeds if s.anchor]
            
            for candidate in all_chunks:
                if candidate.anchor:
                    # Проверяем, является ли этот chunk соседом какого-либо seed
                    for seed_anchor in seed_anchors:
                        # Простая эвристика: если anchor близок по алфавиту/порядку
                        if self._are_anchors_nearby(candidate.anchor, seed_anchor):
                            expanded.append(candidate)
                            break
                
                if len(expanded) >= 4:  # Максимум 4 соседа
                    break
            
            return expanded
    
    def _are_anchors_nearby(self, anchor1: str, anchor2: str) -> bool:
        """
        Проверяет, являются ли два anchor соседями (в пределах одной секции).
        
        Простая эвристика: если anchor имеют общий префикс и отличаются только подпунктом.
        """
        # Извлекаем базовый префикс (например, "§164.512")
        import re
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


class NavigationRetriever(BaseRetriever):
    """
    Ретривер для навигационных вопросов типа "какая часть покрывает X".
    
    Возвращает структурные элементы (Part/Subpart/Section), а не цитаты текста.
    Использует двухслойный алгоритм: rule-based + title-based search.
    """
    
    # Правила для быстрых маршрутов
    RULES = {
        "privacy": {
            "keywords": ["privacy", "disclosure", "phi", "uses and disclosures", "minimum necessary"],
            "part": 164,
            "score": 0.95,
            "explanation": "keyword privacy/disclosure/PHI -> Part 164 (Privacy Rule)"
        },
        "security": {
            "keywords": ["security", "administrative safeguards", "encryption"],
            "part": 164,
            "score": 0.95,
            "explanation": "keyword security/encryption -> Part 164 (Security Rule)"
        },
        "transactions": {
            "keywords": ["transactions", "code sets", "identifiers"],
            "part": 162,
            "score": 0.95,
            "explanation": "keyword transactions/code sets -> Part 162"
        },
        "general": {
            "keywords": ["general provisions", "applicability", "definitions", "part 160"],
            "part": 160,
            "score": 0.95,
            "explanation": "keyword general/applicability/definitions -> Part 160"
        }
    }
    
    def __init__(self, db_connection=None):
        """
        Инициализация навигационного ретривера.
        
        Args:
            db_connection: Подключение к PostgreSQL (если None, создается новое)
        """
        self.db = db_connection or get_db_connection()
    
    async def retrieve(
        self,
        question_embedding: List[float],
        max_results: int = 3,
        question: Optional[str] = None,
        doc_id: Optional[str] = None,
        k_sections: int = 10,
        k_answer: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Поиск структурных элементов для навигационных вопросов.
        
        Args:
            question_embedding: Эмбеддинг вопроса (не используется, но требуется для интерфейса)
            max_results: Количество результатов (по умолчанию 3)
            question: Текст вопроса
            doc_id: ID документа (по умолчанию 'hipaa-reg-2013-03-26')
            k_sections: Количество кандидатов для поиска (по умолчанию 10)
            k_answer: Количество итоговых секций для ответа (по умолчанию 3)
        
        Returns:
            Список словарей с информацией о структурных элементах
        """
        doc_id = doc_id or "hipaa-reg-2013-03-26"
        question = question or ""
        question_lower = question.lower()
        
        logger.info(f"NavigationRetriever: question='{question[:50]}...', k_sections={k_sections}, k_answer={k_answer}")
        
        # Шаг 1: Rule-based поиск
        rule_result = self._rule_based_search(question_lower)
        
        if rule_result and rule_result["rule_score"] >= 0.9:
            # Высокая уверенность - используем правило
            logger.info(f"Rule-based match: {rule_result['explanation']}")
            part = rule_result["part"]
            
            # Шаг 3: Получаем suggested entry points
            suggested_sections = await self._get_suggested_sections(
                part=part,
                doc_id=doc_id,
                limit=k_answer
            )
            
            # Формируем ответ
            results = []
            for section in suggested_sections:
                results.append({
                    "part": part,
                    "subpart": section.get("subpart"),
                    "section_id": section["section_id"],
                    "section_number": section["section_number"],
                    "section_title": section["section_title"],
                    "anchor": section.get("anchor"),
                    "page_start": section.get("page_start"),
                    "page_end": section.get("page_end"),
                    "scores": {
                        "rule_score": rule_result["rule_score"],
                        "title_score": 0.0,
                        "final_score": rule_result["rule_score"]
                    },
                    "explanation": rule_result["explanation"]
                })
            
            return results[:max_results]
        
        # Шаг 2: Title-based поиск
        title_results = await self._title_search(
            question=question,
            doc_id=doc_id,
            limit=k_sections
        )
        
        if not title_results:
            logger.warning("Title search не вернул результатов")
            return []
        
        # Нормализуем title_score
        if title_results:
            max_title_score = max(r["title_score"] for r in title_results)
            if max_title_score > 0:
                for r in title_results:
                    r["title_score_norm"] = r["title_score"] / max_title_score
            else:
                for r in title_results:
                    r["title_score_norm"] = 0.0
        
        # Шаг 4: Scoring и выбор
        rule_score = rule_result["rule_score"] if rule_result else 0.0
        
        for result in title_results:
            title_score_norm = result.get("title_score_norm", 0.0)
            result["final_score"] = max(rule_score, 0.6 * title_score_norm)
        
        # Сортируем по final_score
        title_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Группируем по part и выбираем лучшие секции
        part_sections = {}
        for result in title_results:
            part = result["part"]
            if part not in part_sections:
                part_sections[part] = []
            part_sections[part].append(result)
        
        # Выбираем top-k секций из лучшего part или распределяем по parts
        final_results = []
        seen_sections = set()
        
        for part, sections in sorted(part_sections.items(), key=lambda x: max(s["final_score"] for s in x[1]), reverse=True):
            for section in sections[:k_answer]:
                if section["section_id"] not in seen_sections:
                    final_results.append({
                        "part": section["part"],
                        "subpart": section.get("subpart"),
                        "section_id": section["section_id"],
                        "section_number": section["section_number"],
                        "section_title": section["section_title"],
                        "anchor": section.get("anchor"),
                        "page_start": section.get("page_start"),
                        "page_end": section.get("page_end"),
                        "scores": {
                            "rule_score": rule_score,
                            "title_score": section["title_score"],
                            "final_score": section["final_score"]
                        },
                        "explanation": section.get("explanation", f"title match: {section['section_title']}")
                    })
                    seen_sections.add(section["section_id"])
                    if len(final_results) >= max_results:
                        break
            if len(final_results) >= max_results:
                break
        
        logger.info(f"Найдено навигационных результатов: {len(final_results)}")
        return final_results
    
    def _rule_based_search(self, question_lower: str) -> Optional[Dict[str, Any]]:
        """
        Rule-based поиск - проверка правил для быстрых маршрутов.
        
        Returns:
            Словарь с part, rule_score, explanation или None
        """
        for rule_name, rule_data in self.RULES.items():
            for keyword in rule_data["keywords"]:
                if keyword in question_lower:
                    return {
                        "part": rule_data["part"],
                        "rule_score": rule_data["score"],
                        "explanation": rule_data["explanation"]
                    }
        return None
    
    async def _title_search(
        self,
        question: str,
        doc_id: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Title-based поиск по заголовкам секций (FTS + trigram fallback).
        """
        results = []
        
        # 2A) FTS по заголовкам
        if question and question.strip():
            try:
                with self.db.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            s.section_id,
                            s.part,
                            s.subpart,
                            s.section_number,
                            s.section_title,
                            s.anchor,
                            s.page_start,
                            s.page_end,
                            ts_rank_cd(
                                to_tsvector('english', COALESCE(s.section_title, '') || ' ' || COALESCE(s.section_number, '')),
                                plainto_tsquery('english', %s)
                            ) AS title_score
                        FROM sections s
                        WHERE s.doc_id = %s
                          AND plainto_tsquery('english', %s) @@ to_tsvector('english', COALESCE(s.section_title, '') || ' ' || COALESCE(s.section_number, ''))
                        ORDER BY title_score DESC
                        LIMIT %s
                    """, (question, doc_id, question, limit))
                    
                    rows = cur.fetchall()
                    if rows and len(rows) > 1:
                        for row in rows:
                            results.append({
                                "section_id": row[0],
                                "part": row[1],
                                "subpart": row[2],
                                "section_number": row[3],
                                "section_title": row[4],
                                "anchor": row[5],
                                "page_start": row[6],
                                "page_end": row[7],
                                "title_score": float(row[8]) if row[8] else 0.0,
                                "explanation": f"FTS match: {row[4]}"
                            })
                        return results
            except Exception as e:
                logger.warning(f"FTS поиск по заголовкам не удался: {e}")
                self.db.rollback()
            
        # 2B) Trigram fallback (только если FTS не дал результатов)
        if not results:
            query_hint = self._extract_query_hint(question)
            if query_hint:
                try:
                    with self.db.cursor() as cur:
                        # Проверяем, есть ли расширение pg_trgm
                        cur.execute("""
                            SELECT 
                                s.section_id,
                                s.part,
                                s.subpart,
                                s.section_number,
                                s.section_title,
                                s.anchor,
                                s.page_start,
                                s.page_end,
                                similarity(
                                    COALESCE(s.section_title, '') || ' ' || COALESCE(s.section_number, ''),
                                    %s
                                ) AS title_score
                            FROM sections s
                            WHERE s.doc_id = %s
                              AND similarity(
                                    COALESCE(s.section_title, '') || ' ' || COALESCE(s.section_number, ''),
                                    %s
                                ) > 0.1
                            ORDER BY title_score DESC
                            LIMIT %s
                        """, (query_hint, doc_id, query_hint, limit))
                        
                        rows = cur.fetchall()
                        if rows:
                            for row in rows:
                                results.append({
                                    "section_id": row[0],
                                    "part": row[1],
                                    "subpart": row[2],
                                    "section_number": row[3],
                                    "section_title": row[4],
                                    "anchor": row[5],
                                    "page_start": row[6],
                                    "page_end": row[7],
                                    "title_score": float(row[8]) if row[8] else 0.0,
                                    "explanation": f"trigram match: {row[4]}"
                                })
                            return results
                except Exception as e:
                    logger.warning(f"Trigram поиск не удался (возможно, расширение pg_trgm не установлено): {e}")
                    self.db.rollback()
                    # Продолжаем к fallback
            
        # Fallback: простой поиск по ключевым словам (если еще нет результатов)
        if not results:
            keywords = self._extract_keywords(question)
            if keywords:
                try:
                    with self.db.cursor() as cur:
                        keyword_patterns = [f"%%{kw}%%" for kw in keywords]
                        cur.execute("""
                            SELECT 
                                s.section_id,
                                s.part,
                                s.subpart,
                                s.section_number,
                                s.section_title,
                                s.anchor,
                                s.page_start,
                                s.page_end,
                                CASE 
                                    WHEN s.section_title ILIKE ANY(%s) THEN 1.0
                                    WHEN s.section_number ILIKE ANY(%s) THEN 0.8
                                    ELSE 0.5
                                END AS title_score
                            FROM sections s
                            WHERE s.doc_id = %s
                              AND (s.section_title ILIKE ANY(%s) OR s.section_number ILIKE ANY(%s))
                            ORDER BY title_score DESC, s.section_number
                            LIMIT %s
                        """, (
                            keyword_patterns,
                            keyword_patterns,
                            doc_id,
                            keyword_patterns,
                            keyword_patterns,
                            limit
                        ))
                        
                        rows = cur.fetchall()
                        for row in rows:
                            results.append({
                                "section_id": row[0],
                                "part": row[1],
                                "subpart": row[2],
                                "section_number": row[3],
                                "section_title": row[4],
                                "anchor": row[5],
                                "page_start": row[6],
                                "page_end": row[7],
                                "title_score": float(row[8]) if row[8] else 0.0,
                                "explanation": f"keyword match: {row[4]}"
                            })
                except Exception as e:
                    logger.warning(f"Keyword fallback не удался: {e}")
                    self.db.rollback()
        
        return results
    
    def _extract_query_hint(self, question: str) -> str:
        """
        Извлекает ключевую фразу из вопроса для trigram поиска.
        Удаляет stopwords и оставляет ключевые слова.
        """
        # Простой список stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                     "have", "has", "had", "do", "does", "did", "will", "would", "should",
                     "could", "may", "might", "must", "can", "what", "which", "where",
                     "when", "who", "why", "how", "this", "that", "these", "those",
                     "and", "or", "but", "if", "of", "in", "on", "at", "to", "for",
                     "with", "by", "from", "about", "into", "through", "during"}
        
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return " ".join(keywords[:5])  # Берем первые 5 ключевых слов
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Извлекает ключевые слова из вопроса."""
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                     "have", "has", "had", "do", "does", "did", "will", "would", "should",
                     "could", "may", "might", "must", "can", "what", "which", "where",
                     "when", "who", "why", "how", "this", "that", "these", "those",
                     "and", "or", "but", "if", "of", "in", "on", "at", "to", "for",
                     "with", "by", "from", "about", "into", "through", "during"}
        
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    async def _get_suggested_sections(
        self,
        part: int,
        doc_id: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Получает suggested entry points - топ секции внутри указанного part.
        """
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT 
                    s.section_id,
                    s.part,
                    s.subpart,
                    s.section_number,
                    s.section_title,
                    s.anchor,
                    s.page_start,
                    s.page_end
                FROM sections s
                WHERE s.doc_id = %s
                  AND s.part = %s
                ORDER BY s.section_number
                LIMIT %s
            """, (doc_id, part, limit))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "section_id": row[0],
                    "part": row[1],
                    "subpart": row[2],
                    "section_number": row[3],
                    "section_title": row[4],
                    "anchor": row[5],
                    "page_start": row[6],
                    "page_end": row[7]
                })
            
            return results


def get_retriever_for_category(category: str, question: Optional[str] = None, db_connection=None) -> BaseRetriever:
    """
    Возвращает подходящий ретривер для категории вопроса.
    
    Args:
        category: Категория вопроса
        question: Текст вопроса (для определения навигационных вопросов)
        db_connection: Подключение к базе данных (если None, создается новое)
    
    Returns:
        Экземпляр ретривера
    """
    retriever_map = {
        "overview / purpose": OverviewPurposeRetriever,
        "definition": DefinitionRetriever,
        "scope / applicability": ScopeRetriever,
        "penalties": PenaltiesRetriever,
        "procedural / best practices": ProceduralRetriever,
        "permission / disclosure": PermissionDisclosureRetriever,
        "citation-required": CitationRetriever,
    }
    
    # Для навигационных вопросов (которые спрашивают "which part", "where is") используем NavigationRetriever
    # Определяем по ключевым словам в вопросе
    if question:
        question_lower = question.lower()
        navigation_keywords = ["which part", "where is", "where are", "where does", "which section", "which subpart"]
        if any(kw in question_lower for kw in navigation_keywords):
            return NavigationRetriever(db_connection)
    
    retriever_class = retriever_map.get(category, GeneralRetriever)  # GeneralRetriever как fallback
    return retriever_class(db_connection)
