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
        
        logger.info(f"CitationRetriever (from new module): question='{question[:50]}...', anchor_prefix={anchor_prefix}, k={k}")
        
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
            boost_words = [
                "law enforcement", "police", "court", "warrant",
                "subpoena", "administrative request", "disclosure", "disclose"
            ]
            fts_query = build_fts_query(question, boost_words)
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
