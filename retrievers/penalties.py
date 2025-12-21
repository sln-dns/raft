"""Ретривер для вопросов о санкциях/штрафах."""

from typing import List, Optional, Dict, Any
import logging

from .base import BaseRetriever, CandidateChunk
from .utils import (
    get_db_connection, normalize_fts_scores, merge_candidates, 
    calculate_final_score, build_fts_query, calculate_amount_score
)
from embeddings import get_embedding_client

logger = logging.getLogger(__name__)


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
        
        logger.info(f"PenaltiesRetriever (from new module): question='{question[:50]}...', k={k}, seed_k={seed_k}, expand={expand_section}, need_amounts={need_amounts}")
        
        # Step 0: Подготовка FTS query (усиление "penalty")
        fts_query = build_fts_query(question, self.PENALTY_BOOST_WORDS)
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
            candidate.amount_score = calculate_amount_score(candidate.text_raw)
        
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
        # Нормализуем FTS scores
        normalize_fts_scores(fts_candidates)
        
        # Объединяем кандидатов
        merged_dict = merge_candidates(vector_candidates, fts_candidates)
        
        # Вычисляем base_score для всех (0.65 vector + 0.35 fts)
        for candidate in merged_dict.values():
            candidate.final_score = calculate_final_score(
                candidate,
                vector_weight=0.65,
                fts_weight=0.35
            )
        
        # Сортируем по base_score
        merged_list = sorted(merged_dict.values(), key=lambda x: x.final_score, reverse=True)
        
        return merged_list
    
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
                sibling.amount_score = calculate_amount_score(sibling.text_raw)
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
                  AND c.chunk_id != ALL(%s::text[])
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
                    amount_score = calculate_amount_score(candidate.text_raw)
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
