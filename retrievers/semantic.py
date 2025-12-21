"""Семантический ретривер - поиск по косинусному сходству эмбеддингов."""

from typing import List, Dict, Any
import logging

from .base import BaseRetriever
from .utils import get_db_connection

logger = logging.getLogger(__name__)


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
        Поиск релевантных чанков по косинусному сходству эмбеддингов.
        
        Args:
            question_embedding: Эмбеддинг вопроса (размерность 4096)
            max_results: Максимальное количество результатов
        
        Returns:
            Список словарей с информацией о чанках
        """
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
                    1 - (c.embedding <=> %s::vector) AS similarity
                FROM chunks c
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """, (question_embedding, question_embedding, max_results))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "chunk_id": row[0],
                    "anchor": row[1],
                    "section_id": row[2],
                    "section_number": row[3],
                    "section_title": row[4],
                    "text_raw": row[5],
                    "page_start": row[6],
                    "page_end": row[7],
                    "similarity": float(row[8])
                })
            
            return results
