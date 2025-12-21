"""Ретривер для вопросов типа 'overview / purpose'."""

from typing import List, Optional, Dict, Any
import logging

from .base import BaseRetriever, CandidateChunk
from .utils import get_db_connection, normalize_fts_scores, merge_candidates, calculate_final_score
from embeddings import get_embedding_client

logger = logging.getLogger(__name__)


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
        
        logger.info(f"OverviewPurposeRetriever (from new module): part={part}, max_results={max_results}")
        
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
        # Нормализуем FTS scores
        normalize_fts_scores(fts_candidates)
        
        # Объединяем кандидатов
        merged_dict = merge_candidates(vector_candidates, fts_candidates)
        
        # Вычисляем final_score для всех (0.7 vector + 0.3 fts)
        for candidate in merged_dict.values():
            candidate.final_score = calculate_final_score(
                candidate,
                vector_weight=0.7,
                fts_weight=0.3
            )
        
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
