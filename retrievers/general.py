"""Ретривер для прочих вопросов (General Fallback)."""

from typing import List, Optional, Dict, Any
import logging

from .base import BaseRetriever, CandidateChunk
from .utils import (
    get_db_connection, normalize_fts_scores, merge_candidates,
    calculate_final_score, determine_part_hint
)
from embeddings import get_embedding_client

logger = logging.getLogger(__name__)


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
        
        logger.info(f"GeneralRetriever (from new module): question='{question[:50]}...', k={k}, seed_k={seed_k}, max_per_section={max_per_section}")
        
        # Step 0: Part hint (легкое предположение)
        part_hint = None
        if use_part_hint:
            part_hint = determine_part_hint(question)
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
        normalize_fts_scores(fts_candidates)
        
        # Объединяем кандидатов
        merged_dict = merge_candidates(
            [c for c in candidates if c.vector_score > 0],
            fts_candidates
        )
        
        # Вычисляем final_score для всех
        for candidate in merged_dict.values():
            candidate.final_score = calculate_final_score(
                candidate,
                vector_weight=vector_weight,
                fts_weight=fts_weight
            )
        
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
