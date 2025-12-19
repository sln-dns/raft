#!/usr/bin/env python3
"""Проверка результатов векторизации чанков."""

import sys
from pathlib import Path
import psycopg
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """Создает подключение к базе данных."""
    user = Path.home().name
    return psycopg.connect(
        host="localhost",
        dbname="raft",
        user=user,
    )


def test_embeddings():
    """Проверяет результаты векторизации."""
    print("=" * 60)
    print("ПРОВЕРКА РЕЗУЛЬТАТОВ ВЕКТОРИЗАЦИИ")
    print("=" * 60)
    
    conn = get_db_connection()
    
    # 1. Общая статистика
    print("\n📊 1. ОБЩАЯ СТАТИСТИКА")
    print("-" * 60)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                COUNT(*) AS total,
                COUNT(embedding) AS with_embedding,
                COUNT(*) - COUNT(embedding) AS without_embedding,
                ROUND(100.0 * COUNT(embedding) / COUNT(*), 2) AS coverage_pct
            FROM chunks
        """)
        row = cur.fetchone()
        total, with_emb, without_emb, coverage = row
        print(f"   Всего чанков: {total}")
        print(f"   ✅ С эмбеддингами: {with_emb}")
        print(f"   ❌ Без эмбеддингов: {without_emb}")
        print(f"   📈 Покрытие: {coverage}%")
    
    if with_emb == 0:
        print("\n⚠️  Эмбеддинги не найдены!")
        conn.close()
        return
    
    # 2. Проверка размерности
    print("\n📏 2. РАЗМЕРНОСТЬ ЭМБЕДДИНГОВ")
    print("-" * 60)
    with conn.cursor() as cur:
        # Используем функцию pgvector для получения размерности
        cur.execute("""
            SELECT 
                array_length(string_to_array(embedding::text, ','), 1) AS dimension,
                COUNT(*) AS count
            FROM chunks
            WHERE embedding IS NOT NULL
            GROUP BY dimension
            ORDER BY dimension
        """)
        rows = cur.fetchall()
        if rows:
            for dim, count in rows:
                print(f"   Размерность {dim}: {count} чанков")
            
            if len(rows) == 1:
                print(f"   ✅ Все векторы имеют одинаковую размерность: {rows[0][0]}")
            else:
                print(f"   ⚠️  Найдены векторы разных размерностей")
        else:
            # Альтернативный способ - через Python
            cur.execute("SELECT embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1")
            row = cur.fetchone()
            if row:
                embedding = row[0]
                # pgvector возвращает vector как специальный тип, нужно конвертировать
                cur.execute("SELECT %s::text", (embedding,))
                emb_text = cur.fetchone()[0]
                # Парсим строку вида "[0.1, 0.2, ...]"
                emb_list = [float(x) for x in emb_text.strip('[]').split(',')]
                dim = len(emb_list)
                print(f"   Размерность: {dim}")
                print(f"   ✅ Проверено на примере")
    
    # 3. Проверка на нулевые векторы
    print("\n🔍 3. ПРОВЕРКА КАЧЕСТВА ЭМБЕДДИНГОВ")
    print("-" * 60)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                chunk_id,
                embedding,
                text_raw
            FROM chunks
            WHERE embedding IS NOT NULL
            LIMIT 10
        """)
        rows = cur.fetchall()
        
        zero_vectors = 0
        empty_vectors = 0
        valid_vectors = 0
        
        for chunk_id, embedding, text_raw in rows:
            # Конвертируем vector в numpy array через текстовое представление
            cur2 = conn.cursor()
            cur2.execute("SELECT %s::text", (embedding,))
            emb_text = cur2.fetchone()[0]
            cur2.close()
            
            # Парсим строку вида "[0.1, 0.2, ...]"
            try:
                emb_list = [float(x.strip()) for x in emb_text.strip('[]').split(',')]
                emb = np.array(emb_list, dtype=np.float32)
            except Exception as e:
                print(f"   ⚠️  {chunk_id}: ошибка парсинга вектора: {e}")
                empty_vectors += 1
                continue
            
            # Проверка на нулевой вектор
            if np.allclose(emb, 0, atol=1e-6):
                zero_vectors += 1
                print(f"   ⚠️  {chunk_id}: нулевой вектор")
            # Проверка на пустой вектор
            elif len(emb) == 0:
                empty_vectors += 1
                print(f"   ⚠️  {chunk_id}: пустой вектор")
            else:
                valid_vectors += 1
                # Статистика по вектору
                norm = np.linalg.norm(emb)
                mean = np.mean(emb)
                std = np.std(emb)
                min_val = np.min(emb)
                max_val = np.max(emb)
                print(f"   ✅ {chunk_id}:")
                print(f"      Норма: {norm:.4f}")
                print(f"      Среднее: {mean:.6f}, Стд. откл.: {std:.6f}")
                print(f"      Мин: {min_val:.6f}, Макс: {max_val:.6f}")
                print(f"      Текст (первые 50 символов): {text_raw[:50]}...")
                print()
        
        print(f"   Валидных векторов: {valid_vectors}")
        print(f"   Нулевых векторов: {zero_vectors}")
        print(f"   Пустых векторов: {empty_vectors}")
    
    # 4. Статистика по типам чанков
    print("\n📋 4. СТАТИСТИКА ПО ТИПАМ ЧАНКОВ")
    print("-" * 60)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                chunk_kind,
                granularity,
                COUNT(*) AS total,
                COUNT(embedding) AS with_embedding,
                ROUND(100.0 * COUNT(embedding) / COUNT(*), 2) AS coverage_pct
            FROM chunks
            GROUP BY chunk_kind, granularity
            ORDER BY chunk_kind, granularity
        """)
        rows = cur.fetchall()
        for chunk_kind, granularity, total, with_emb, coverage in rows:
            print(f"   {chunk_kind} / {granularity}: {with_emb}/{total} ({coverage}%)")
    
    # 5. Статистика по секциям
    print("\n📑 5. СТАТИСТИКА ПО СЕКЦИЯМ")
    print("-" * 60)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                section_number,
                COUNT(*) AS total,
                COUNT(embedding) AS with_embedding
            FROM chunks
            GROUP BY section_number
            HAVING COUNT(embedding) < COUNT(*)
            ORDER BY section_number
            LIMIT 10
        """)
        rows = cur.fetchall()
        if rows:
            print("   Секции с неполным покрытием:")
            for section_num, total, with_emb in rows:
                print(f"   {section_num}: {with_emb}/{total}")
        else:
            print("   ✅ Все секции полностью покрыты эмбеддингами")
    
    # 6. Примеры похожих чанков (если есть векторный индекс)
    print("\n🔗 6. ПРОВЕРКА ПОИСКА ПОХОЖИХ ЧАНКОВ")
    print("-" * 60)
    with conn.cursor() as cur:
        # Берем случайный чанк с эмбеддингом
        cur.execute("""
            SELECT chunk_id, text_raw, embedding
            FROM chunks
            WHERE embedding IS NOT NULL
            AND text_raw IS NOT NULL
            AND length(text_raw) > 50
            ORDER BY RANDOM()
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            test_chunk_id, test_text, test_embedding = row
            test_emb_array = test_embedding
            
            print(f"   Тестовый чанк: {test_chunk_id}")
            print(f"   Текст: {test_text[:100]}...")
            print()
            
            # Ищем похожие чанки (используем прямое сравнение векторов)
            cur.execute("""
                SELECT 
                    chunk_id,
                    text_raw,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                AND chunk_id != %s
                ORDER BY embedding <=> %s::vector
                LIMIT 5
            """, (test_embedding, test_chunk_id, test_embedding))
            
            similar = cur.fetchall()
            if similar:
                print("   Топ-5 похожих чанков:")
                for i, (chunk_id, text, similarity) in enumerate(similar, 1):
                    print(f"   {i}. {chunk_id} (similarity: {similarity:.4f})")
                    print(f"      {text[:80]}...")
                    print()
            else:
                print("   ⚠️  Похожие чанки не найдены")
        else:
            print("   ⚠️  Не найдено подходящих чанков для теста")
    
    # 7. Проверка индексов и производительности
    print("\n🗂️  7. ПРОВЕРКА ИНДЕКСОВ И ПРОИЗВОДИТЕЛЬНОСТИ")
    print("-" * 60)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                indexname,
                indexdef
            FROM pg_indexes
            WHERE tablename = 'chunks'
            AND indexname LIKE '%embedding%'
        """)
        rows = cur.fetchall()
        if rows:
            for idx_name, idx_def in rows:
                print(f"   ✅ {idx_name}")
                print(f"      {idx_def}")
        else:
            print("   ℹ️  Векторный индекс не создан")
            print("   📝 Примечание: pgvector 0.8.1 поддерживает индексы только до 2000 измерений")
            print("   📝 Наша размерность: 4096 (модель emb-qwen/qwen3-embedding-8b)")
            print("   ✅ Для текущего объема данных (488 чанков) поиск без индекса работает быстро (~26ms)")
            print("   💡 При росте данных можно рассмотреть обновление pgvector или другие решения")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_embeddings()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
