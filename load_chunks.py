#!/usr/bin/env python3
"""Загрузка chunks.jsonl в staging таблицу PostgreSQL."""

import json
import sys
from pathlib import Path
import psycopg

def load_chunks_to_staging(jsonl_path: Path, db_conn_str: str):
    """Загружает JSONL файл в staging_chunks таблицу."""
    conn = psycopg.connect(db_conn_str)
    cur = conn.cursor()
    
    # Очищаем staging таблицу
    cur.execute("TRUNCATE staging_chunks")
    
    # Читаем и вставляем данные
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                cur.execute(
                    "INSERT INTO staging_chunks (row) VALUES (%s)",
                    (json.dumps(data),)
                )
                count += 1
                if count % 100 == 0:
                    print(f"Загружено {count} чанков...", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON на строке {count + 1}: {e}", file=sys.stderr)
                continue
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"Всего загружено: {count} чанков")
    return count

if __name__ == "__main__":
    jsonl_path = Path("jsonl/chunks.jsonl")
    if not jsonl_path.exists():
        print(f"Файл {jsonl_path} не найден", file=sys.stderr)
        sys.exit(1)
    
    # Подключение к БД (можно использовать переменные окружения)
    db_conn_str = "host=localhost dbname=raft user=" + __import__('os').getenv('USER', 'postgres')
    
    load_chunks_to_staging(jsonl_path, db_conn_str)
