"""Ретривер для вопросов типа 'scope / applicability'."""

from typing import List, Optional, Dict, Any
import logging

from .base import BaseRetriever, CandidateChunk
from .utils import get_db_connection, normalize_fts_scores, merge_candidates, calculate_final_score, build_fts_query
from embeddings import get_embedding_client

logger = logging.getLogger(__name__)


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
        
        logger.info(f"ScopeRetriever (from new module): question='{question[:50]}...', k={k}, seed_k={seed_k}, expand={expand_with_siblings}")
        
        # Step 0: Подготовка FTS query (усиление)
        fts_query = build_fts_query(question, self.APPLICABILITY_BOOST_WORDS)
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
        # Нормализуем FTS scores
        normalize_fts_scores(fts_seeds)
        
        # Объединяем кандидатов
        merged_dict = merge_candidates(vector_seeds, fts_seeds)
        
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
                  AND c.chunk_id != ALL(%s::text[])
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
                  AND c.chunk_id != ALL(%s::text[])
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
