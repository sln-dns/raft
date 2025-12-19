#!/usr/bin/env python3
"""Генерация и сохранение эмбеддингов для всех чанков в базе данных."""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import psycopg
from embeddings import get_embedding_client
from dotenv import load_dotenv

load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('generate_embeddings.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Создает подключение к базе данных."""
    user = Path.home().name
    return psycopg.connect(
        host="localhost",
        dbname="raft",
        user=user,
    )


def get_chunks_without_embeddings(conn):
    """Получает список чанков без эмбеддингов."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_id, text_raw
            FROM chunks
            WHERE embedding IS NULL
            ORDER BY chunk_id
        """)
        return cur.fetchall()


def update_chunk_embedding(conn, chunk_id: str, embedding: list):
    """Обновляет эмбеддинг для чанка."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE chunks
            SET embedding = %s::vector
            WHERE chunk_id = %s
        """, (embedding, chunk_id))
    conn.commit()


def generate_embeddings_for_all_chunks():
    """Генерирует эмбеддинги для всех чанков без эмбеддингов."""
    logger.info("=" * 60)
    logger.info("Начало генерации эмбеддингов")
    logger.info("=" * 60)
    
    logger.info("🔌 Подключение к базе данных...")
    conn = get_db_connection()
    logger.info("✅ Подключение установлено")
    
    logger.info("📊 Получение списка чанков без эмбеддингов...")
    chunks = get_chunks_without_embeddings(conn)
    total = len(chunks)
    logger.info(f"📝 Найдено {total} чанков для векторизации")
    
    if total == 0:
        logger.info("✅ Все чанки уже имеют эмбеддинги!")
        conn.close()
        return
    
    logger.info("🚀 Начинаем генерацию эмбеддингов...")
    logger.info(f"⏱️  Ожидаемое время выполнения: ~{total} секунд (минимум)")
    logger.info("-" * 60)
    
    client = get_embedding_client()
    logger.info(f"🤖 Используется модель: {client.model}")
    logger.info(f"📏 Размерность эмбеддингов: {client.get_embedding_dimension()}")
    logger.info("-" * 60)
    
    success_count = 0
    error_count = 0
    start_time = time.time()
    last_request_time = 0
    
    for idx, (chunk_id, text_raw) in enumerate(chunks, 1):
        try:
            # Показываем прогресс
            text_preview = text_raw[:50].replace('\n', ' ') + "..." if len(text_raw) > 50 else text_raw.replace('\n', ' ')
            logger.info(f"[{idx}/{total}] Обработка {chunk_id}")
            logger.debug(f"  Текст (первые 50 символов): {text_preview}")
            logger.debug(f"  Длина текста: {len(text_raw)} символов")
            
            # Проверяем, что текст не пустой
            if not text_raw or not text_raw.strip():
                logger.warning(f"  ⚠️  Пропущен (пустой текст)")
                error_count += 1
                continue
            
            # Обеспечиваем задержку минимум 1 секунда между запросами
            current_time = time.time()
            time_since_last = current_time - last_request_time
            if time_since_last < 1.0:
                sleep_time = 1.0 - time_since_last
                logger.debug(f"  ⏳ Ожидание {sleep_time:.2f}с перед запросом...")
                time.sleep(sleep_time)
            
            # Создаем эмбеддинг (последовательно, ждем ответа)
            logger.info(f"  📡 Отправка запроса к API...")
            request_start = time.time()
            embedding = client.create_embedding(text_raw)
            request_duration = time.time() - request_start
            last_request_time = time.time()
            
            logger.info(f"  ✅ Эмбеддинг получен ({request_duration:.2f}с, размерность: {len(embedding)})")
            
            # Обновляем в базе данных
            logger.debug(f"  💾 Сохранение в базу данных...")
            db_start = time.time()
            update_chunk_embedding(conn, chunk_id, embedding)
            db_duration = time.time() - db_start
            logger.info(f"  ✅ Сохранено в БД ({db_duration:.3f}с)")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"  ❌ Ошибка при обработке {chunk_id}: {e}", exc_info=True)
            error_count += 1
            # При ошибке делаем паузу перед следующим запросом
            logger.info(f"  ⏸️  Пауза 1 секунда перед следующим запросом...")
            time.sleep(1)
            continue
        
        # Показываем статистику каждые 10 чанков
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (total - idx) / rate if rate > 0 else 0
            progress_pct = idx * 100 // total
            logger.info("-" * 60)
            logger.info(f"📊 ПРОГРЕСС: {idx}/{total} ({progress_pct}%)")
            logger.info(f"   ✅ Успешно: {success_count}")
            logger.info(f"   ❌ Ошибок: {error_count}")
            logger.info(f"   ⚡ Скорость: {rate:.2f} чанков/сек")
            logger.info(f"   ⏱️  Прошло: {elapsed:.0f} сек")
            logger.info(f"   ⏳ Осталось: ~{remaining:.0f} сек (~{remaining/60:.1f} мин)")
            logger.info("-" * 60)
    
    conn.close()
    
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info(f"📊 Статистика:")
    logger.info(f"   Всего обработано: {total}")
    logger.info(f"   ✅ Успешно: {success_count}")
    logger.info(f"   ❌ Ошибок: {error_count}")
    logger.info(f"   ⏱️  Общее время: {elapsed:.1f} сек ({elapsed/60:.1f} мин)")
    if elapsed > 0:
        logger.info(f"   ⚡ Средняя скорость: {total/elapsed:.2f} чанков/сек")
        logger.info(f"   📈 Фактическая скорость: {success_count/elapsed:.2f} успешных/сек")
    logger.info("=" * 60)
    logger.info(f"📝 Логи сохранены в файл: generate_embeddings.log")


if __name__ == "__main__":
    try:
        generate_embeddings_for_all_chunks()
    except KeyboardInterrupt:
        print("\n\n⚠️  Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}", file=sys.stderr)
        sys.exit(1)
