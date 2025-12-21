"""Ретривер для вопросов типа 'permission / disclosure'."""

from typing import List, Optional, Dict, Any, Tuple
import logging

from .base import BaseRetriever, CandidateChunk
from .utils import (
    get_db_connection, normalize_fts_scores, merge_candidates,
    calculate_final_score, build_fts_query, calculate_clause_score_disclosure
)
from embeddings import get_embedding_client

logger = logging.getLogger(__name__)


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
        
        logger.info(f"PermissionDisclosureRetriever (from new module): question='{question[:50]}...', k={k}, seed_k={seed_k}, part={part}")
        
        # Step 0: Topic extraction (кому/куда раскрытие)
        topic, topic_tokens = self._extract_disclosure_topic(question)
        logger.info(f"Извлеченная тема раскрытия: {topic}, токены: {topic_tokens[:5]}...")
        
        # Step 1: FTS query (disclosure-focused)
        fts_query = build_fts_query(question, self.DISCLOSURE_TOKENS, topic_tokens)
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
            candidate.keyword_score = calculate_clause_score_disclosure(candidate.text_raw, topic_tokens)
        
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
    
    def _extract_disclosure_topic(self, question: str) -> Tuple[Optional[str], List[str]]:
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
        # Нормализуем FTS scores
        normalize_fts_scores(fts_candidates)
        
        # Объединяем кандидатов
        merged_dict = merge_candidates(vector_candidates, fts_candidates)
        
        return list(merged_dict.values())
    
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
                    candidate.keyword_score = calculate_clause_score_disclosure(candidate.text_raw, topic_tokens)
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
                sibling.keyword_score = calculate_clause_score_disclosure(sibling.text_raw, topic_tokens)
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
