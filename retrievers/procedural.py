"""Ретривер для вопросов типа 'procedural / best practice'."""

from typing import List, Optional, Dict, Any, Tuple
import logging

from .base import BaseRetriever, CandidateChunk
from .utils import (
    get_db_connection, normalize_fts_scores, merge_candidates,
    calculate_final_score, build_fts_query, calculate_keyword_score_procedural
)
from embeddings import get_embedding_client

logger = logging.getLogger(__name__)


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
        "minimum_necessary": [
            "minimum necessary",
            "minimum necessary standard",
            "reasonable efforts",
            "limit",
            "accomplish the intended purpose",
            "minimum amount",
            "reasonably necessary",
        ],
    }
    
    # Soft boost anchors для regulatory principles (например, minimum necessary)
    REGULATORY_PRINCIPLE_BOOST_ANCHORS = {
        "minimum_necessary": ["§164.502", "§164.514"],  # §164.502(b), §164.514(d) и связанные
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
        category: Optional[str] = None,
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
        
        logger.info(f"ProceduralRetriever (from new module): question='{question[:50]}...', k={k}, seed_k={seed_k}, expand={expand_section}, need_yesno={need_yesno}, category={category}")
        
        # Step 0: Определить под-тему и ключевые токены
        topic, topic_tokens = self._extract_topic_and_tokens(question, category)
        logger.info(f"Извлеченная тема: {topic}, токены: {topic_tokens[:5]}...")
        
        # Step 1: Подготовка FTS query (усиление)
        fts_query = build_fts_query(question, self.SECURITY_BOOST_WORDS, topic_tokens)
        
        # Дополнительное усиление FTS для regulatory_principle (minimum necessary)
        if category == "regulatory_principle" and topic == "minimum_necessary":
            # Добавляем специфичные фразы для minimum necessary
            minimum_necessary_phrases = [
                "minimum necessary",
                "reasonable efforts",
                "intended purpose"
            ]
            fts_query = f"{fts_query} {' '.join(minimum_necessary_phrases)}"
            logger.info(f"FTS query усилен для minimum_necessary: {fts_query[:150]}...")
        else:
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
            candidate.keyword_score = calculate_keyword_score_procedural(candidate.text_raw, topic_tokens)
        
        # Step 3.5: Hard boost для regulatory principles (если категория regulatory_principle)
        if category == "regulatory_principle" and topic in self.REGULATORY_PRINCIPLE_BOOST_ANCHORS:
            boost_anchors = self.REGULATORY_PRINCIPLE_BOOST_ANCHORS[topic]
            for candidate in candidates:
                if candidate.anchor:
                    # Проверяем, начинается ли anchor с boost anchor prefix
                    for boost_anchor in boost_anchors:
                        if candidate.anchor.startswith(boost_anchor):
                            # Hard boost: увеличиваем final_score на 20% (1.2x)
                            candidate.anchor_boost = 1.2
                            logger.info(f"Applied hard boost (1.2x) to anchor {candidate.anchor} for topic {topic}")
                            break
        
        # Step 4: Merge + final scoring
        for candidate in candidates:
            # Нормализация уже сделана в _merge_and_score_procedural
            base_final_score = 0.55 * candidate.vector_score + 0.20 * candidate.fts_score + 0.25 * candidate.keyword_score
            
            # Применяем anchor boost если есть
            if hasattr(candidate, 'anchor_boost'):
                candidate.final_score = base_final_score * candidate.anchor_boost
            else:
                candidate.final_score = base_final_score
        
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
    
    def _extract_topic_and_tokens(self, question: str, category: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
        """
        Извлекает под-тему и ключевые токены из вопроса.
        
        Args:
            question: Текст вопроса
            category: Категория вопроса (например, "regulatory_principle")
        
        Returns:
            (topic, topic_tokens) - например ("encryption", ["encrypt", "encryption", ...])
        """
        if not question:
            return None, []
        
        question_lower = question.lower()
        topic_tokens = []
        topic = None
        
        # Если категория regulatory_principle, проверяем сначала minimum_necessary
        if category == "regulatory_principle":
            # Проверяем minimum necessary токены
            if any(token in question_lower for token in self.TOPIC_TOKENS.get("minimum_necessary", [])):
                topic = "minimum_necessary"
                topic_tokens.extend(self.TOPIC_TOKENS["minimum_necessary"])
        
        # Проверяем другие известные темы (если еще не нашли)
        if not topic:
            for topic_name, tokens in self.TOPIC_TOKENS.items():
                if topic_name != "minimum_necessary" and any(token in question_lower for token in tokens):
                    topic = topic_name
                    topic_tokens.extend(tokens)
                    break
        
        # Добавляем общие токены требований
        topic_tokens.extend(self.REQUIREMENT_TOKENS)
        
        # Добавляем общие security-токены
        topic_tokens.extend(self.SECURITY_BOOST_WORDS)
        
        return topic, list(set(topic_tokens))  # Дедупликация
    
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
        # Нормализуем FTS scores
        normalize_fts_scores(fts_candidates)
        
        # Объединяем кандидатов
        merged_dict = merge_candidates(vector_candidates, fts_candidates)
        
        return list(merged_dict.values())
    
    def _select_seeds(
        self, 
        candidates: List[CandidateChunk], 
        seed_k: int,
        category: Optional[str] = None,
        topic: Optional[str] = None,
        boost_anchors: Optional[List[str]] = None
    ) -> List[CandidateChunk]:
        """
        Выбирает seed_k лучших seeds с дедупом по section_id.
        
        Для regulatory_principle с minimum_necessary: гарантирует минимум 1 seed из boost anchors.
        """
        seen_sections = set()
        selected = []
        boost_anchor_selected = False
        
        # Сначала пытаемся выбрать обычным способом
        for candidate in candidates:
            if candidate.section_id not in seen_sections:
                # Проверяем, является ли это boost anchor (для regulatory_principle)
                if category == "regulatory_principle" and boost_anchors and candidate.anchor:
                    for boost_anchor in boost_anchors:
                        if candidate.anchor.startswith(boost_anchor):
                            boost_anchor_selected = True
                            logger.info(f"Selected boost anchor seed: {candidate.anchor}")
                            break
                
                selected.append(candidate)
                seen_sections.add(candidate.section_id)
                if len(selected) >= seed_k:
                    break
        
        # Если не нашли boost anchor в топ-N, принудительно добавляем первый найденный
        if category == "regulatory_principle" and boost_anchors and not boost_anchor_selected:
            logger.warning(f"No boost anchor found in top candidates, forcing selection from boost anchors")
            for candidate in candidates:
                if candidate.anchor:
                    for boost_anchor in boost_anchors:
                        if candidate.anchor.startswith(boost_anchor):
                            # Проверяем, что еще не добавлен
                            if candidate.chunk_id not in {s.chunk_id for s in selected}:
                                # Вставляем в начало списка (приоритет)
                                selected.insert(0, candidate)
                                seen_sections.add(candidate.section_id)
                                boost_anchor_selected = True
                                logger.info(f"Force-selected boost anchor seed: {candidate.anchor}")
                                # Ограничиваем до seed_k
                                if len(selected) > seed_k:
                                    selected = selected[:seed_k]
                                break
                    if boost_anchor_selected:
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
                sibling.keyword_score = calculate_keyword_score_procedural(sibling.text_raw, topic_tokens)
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
                keyword_score = calculate_keyword_score_procedural(candidate.text_raw, topic_tokens)
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
    ) -> Tuple[str, str]:
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
                if any(token in topic_tokens for token in encryption_tokens):
                    anchor_str = result.anchor or "relevant section"
                    return "yes", f"Found explicit term 'encryption' in {anchor_str}"
        
        # Проверка на safeguards/implementation specification (unclear)
        if re.search(r'\b(safeguard|safeguards|implementation specification)\b', text_combined):
            anchor_str = anchors[0] if anchors else "relevant section"
            return "unclear", f"Found safeguards/implementation specifications in {anchor_str}, but no explicit encryption mention"
        
        return "unclear", "Found relevant security content, but unclear if encryption is explicitly required"
