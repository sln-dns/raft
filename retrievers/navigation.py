"""Ретривер для навигационных вопросов типа "какая часть покрывает X"."""

from typing import List, Optional, Dict, Any
import logging
import re

from .base import BaseRetriever
from .utils import get_db_connection

logger = logging.getLogger(__name__)


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
        
        logger.info(f"NavigationRetriever (from new module): question='{question[:50]}...', k_sections={k_sections}, k_answer={k_answer}")
        
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
