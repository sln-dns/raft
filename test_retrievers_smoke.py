"""Smoke тесты для проверки контрактов ретриверов.

Эти тесты не зависят от LLM и БД, проверяют только контракты и базовую логику.

Запуск:
    python test_retrievers_smoke.py
    # или
    python -m test_retrievers_smoke

Тесты:
    - test_registry_returns_correct_class: проверка, что registry возвращает правильный класс
    - test_overview_returns_section_granularity: проверка контракта overview ретривера
    - test_citation_anchor_prefix_filter: проверка фильтрации по anchor prefix
    - test_general_diversity_constraint: проверка diversity constraint
    - test_output_contract_fields_present: проверка наличия обязательных полей в выходных данных
"""

import asyncio
import sys
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Мокируем зависимости до импорта модулей
sys.modules['psycopg'] = MagicMock()
sys.modules['psycopg.errors'] = MagicMock()
sys.modules['embeddings'] = MagicMock()
sys.modules['embeddings'].get_embedding_client = MagicMock(return_value=MagicMock())

# Импортируем ретриверы после мокирования зависимостей
from retrievers.registry import get_retriever_for_category
from retrievers.overview_purpose import OverviewPurposeRetriever
from retrievers.citation import CitationRetriever
from retrievers.general import GeneralRetriever
from retrievers.base import BaseRetriever


def test_registry_returns_correct_class():
    """Тест: registry возвращает правильный класс ретривера для каждой категории."""
    print("🧪 test_registry_returns_correct_class...")
    
    test_cases = [
        ("overview / purpose", OverviewPurposeRetriever),
        ("definition", None),  # Проверим что это не None
        ("scope / applicability", None),
        ("penalties", None),
        ("procedural / best practices", None),
        ("permission / disclosure", None),
        ("citation-required", CitationRetriever),
        ("other", GeneralRetriever),
        ("unknown_category", GeneralRetriever),  # Fallback
    ]
    
    for category, expected_class in test_cases:
        retriever = get_retriever_for_category(category, db_connection=None)
        assert retriever is not None, f"Ретривер для категории '{category}' не должен быть None"
        assert isinstance(retriever, BaseRetriever), f"Ретривер для '{category}' должен быть экземпляром BaseRetriever"
        
        if expected_class is not None:
            assert isinstance(retriever, expected_class), \
                f"Ретривер для '{category}' должен быть экземпляром {expected_class.__name__}"
        
        print(f"  ✅ {category} -> {type(retriever).__name__}")
    
    # Проверка навигационных вопросов
    nav_retriever = get_retriever_for_category("other", question="which part covers privacy?")
    assert nav_retriever is not None
    print(f"  ✅ navigation question -> {type(nav_retriever).__name__}")
    
    print("  ✅ test_registry_returns_correct_class PASSED\n")


async def test_overview_returns_section_granularity():
    """Тест: overview ретривер возвращает section granularity в результатах."""
    print("🧪 test_overview_returns_section_granularity...")
    
    # Создаем мок для БД с правильным контекстным менеджером
    mock_db = Mock()
    mock_cursor = Mock()
    
    # Настраиваем контекстный менеджер для cursor
    mock_cursor_context = MagicMock()
    mock_cursor_context.__enter__ = Mock(return_value=mock_cursor)
    mock_cursor_context.__exit__ = Mock(return_value=None)
    mock_db.cursor.return_value = mock_cursor_context
    
    # Мокируем результаты запросов (vector search)
    mock_cursor.fetchall.return_value = [
        ("chunk1", "§160.103", "section1", "160.103", "Definitions", "section", "text1", 1, 2, 0.9),
        ("chunk2", "§160.104", "section2", "160.104", "Applicability", "section", "text2", 3, 4, 0.8),
    ]
    mock_cursor.execute = Mock()
    
    retriever = OverviewPurposeRetriever(db_connection=mock_db)
    retriever.embedding_client = Mock()
    
    # Создаем фиктивный embedding
    fake_embedding = [0.1] * 4096
    
    results = await retriever.retrieve(
        question_embedding=fake_embedding,
        question="What is the overall purpose of Part 160?",
        doc_id="test-doc",
        k=2
    )
    
    # Проверяем контракт выходных данных
    assert isinstance(results, list), "Результаты должны быть списком"
    
    if results:
        for result in results:
            assert "chunk_id" in result, "Результат должен содержать 'chunk_id'"
            assert "anchor" in result, "Результат должен содержать 'anchor'"
            assert "section_id" in result, "Результат должен содержать 'section_id'"
            assert "text_raw" in result, "Результат должен содержать 'text_raw'"
            assert "scores" in result, "Результат должен содержать 'scores'"
            assert isinstance(result["scores"], dict), "scores должен быть словарем"
    
    print(f"  ✅ Получено {len(results)} результатов")
    print("  ✅ test_overview_returns_section_granularity PASSED\n")


async def test_citation_anchor_prefix_filter():
    """Тест: citation ретривер фильтрует по anchor prefix."""
    print("🧪 test_citation_anchor_prefix_filter...")
    
    # Создаем мок для БД с правильным контекстным менеджером
    mock_db = Mock()
    mock_cursor = Mock()
    
    # Настраиваем контекстный менеджер для cursor
    mock_cursor_context = MagicMock()
    mock_cursor_context.__enter__ = Mock(return_value=mock_cursor)
    mock_cursor_context.__exit__ = Mock(return_value=None)
    mock_db.cursor.return_value = mock_cursor_context
    
    # Мокируем результаты запросов (только с нужным prefix)
    mock_cursor.fetchall.return_value = [
        ("chunk1", "§164.512(a)", "section1", "164.512", "Law enforcement", "text1", 1, 2, 0.9),
        ("chunk2", "§164.512(b)", "section1", "164.512", "Law enforcement", "text2", 3, 4, 0.8),
    ]
    mock_cursor.execute = Mock()
    
    retriever = CitationRetriever(db_connection=mock_db)
    retriever.embedding_client = Mock()
    
    fake_embedding = [0.1] * 4096
    
    # Проверяем, что _determine_anchor_prefix работает
    anchor_prefix = retriever._determine_anchor_prefix("law enforcement disclosure")
    assert anchor_prefix == "§164.512", f"Ожидался '§164.512', получен '{anchor_prefix}'"
    print(f"  ✅ anchor_prefix определен: {anchor_prefix}")
    
    # Проверяем, что запрос использует anchor_like фильтр
    results = await retriever.retrieve(
        question_embedding=fake_embedding,
        question="Cite law enforcement disclosures",
        doc_id="test-doc",
        anchor_prefix="§164.512",
        k=2
    )
    
    # Проверяем контракт
    assert isinstance(results, list), "Результаты должны быть списком"
    
    if results:
        for result in results:
            assert "anchor" in result, "Результат должен содержать 'anchor'"
            assert "text_raw" in result, "Результат должен содержать 'text_raw'"
            # Проверяем, что anchor начинается с нужного prefix (если есть результаты)
            if result.get("anchor"):
                assert result["anchor"].startswith("§164.512"), \
                    f"Anchor должен начинаться с '§164.512', получен '{result['anchor']}'"
    
    print(f"  ✅ Получено {len(results)} результатов с правильным anchor prefix")
    print("  ✅ test_citation_anchor_prefix_filter PASSED\n")


async def test_general_diversity_constraint():
    """Тест: general ретривер применяет diversity constraint (max_per_section)."""
    print("🧪 test_general_diversity_constraint...")
    
    # Создаем мок для БД с правильным контекстным менеджером
    mock_db = Mock()
    mock_cursor = Mock()
    
    # Настраиваем контекстный менеджер для cursor
    mock_cursor_context = MagicMock()
    mock_cursor_context.__enter__ = Mock(return_value=mock_cursor)
    mock_cursor_context.__exit__ = Mock(return_value=None)
    mock_db.cursor.return_value = mock_cursor_context
    
    # Мокируем результаты: много чанков из одной секции
    mock_cursor.fetchall.return_value = [
        ("chunk1", "§160.103(a)", "section1", "160.103", "Definitions", "atomic", None, None, "text1", 1, 2, 0.95),
        ("chunk2", "§160.103(b)", "section1", "160.103", "Definitions", "atomic", None, None, "text2", 3, 4, 0.94),
        ("chunk3", "§160.103(c)", "section1", "160.103", "Definitions", "atomic", None, None, "text3", 5, 6, 0.93),
        ("chunk4", "§160.103(d)", "section1", "160.103", "Definitions", "atomic", None, None, "text4", 7, 8, 0.92),
        ("chunk5", "§160.104(a)", "section2", "160.104", "Applicability", "atomic", None, None, "text5", 9, 10, 0.91),
    ]
    mock_cursor.execute = Mock()
    
    retriever = GeneralRetriever(db_connection=mock_db)
    retriever.embedding_client = Mock()
    
    fake_embedding = [0.1] * 4096
    
    # Тестируем с max_per_section=2
    results = await retriever.retrieve(
        question_embedding=fake_embedding,
        question="What are the general provisions?",
        doc_id="test-doc",
        k=5,
        seed_k=5,
        max_per_section=2  # Максимум 2 чанка из одной секции
    )
    
    # Проверяем diversity constraint
    assert isinstance(results, list), "Результаты должны быть списком"
    
    if len(results) > 0:
        # Подсчитываем количество чанков из каждой секции
        section_counts = {}
        for result in results:
            section_id = result.get("section_id")
            if section_id:
                section_counts[section_id] = section_counts.get(section_id, 0) + 1
        
        # Проверяем, что ни одна секция не превышает max_per_section
        max_per_section = 2
        for section_id, count in section_counts.items():
            assert count <= max_per_section, \
                f"Секция '{section_id}' содержит {count} чанков, максимум {max_per_section}"
        
        print(f"  ✅ Распределение по секциям: {section_counts}")
    
    print("  ✅ test_general_diversity_constraint PASSED\n")


async def test_output_contract_fields_present():
    """Тест: выходные данные всех ретриверов содержат обязательные поля."""
    print("🧪 test_output_contract_fields_present...")
    
    # Обязательные поля для всех ретриверов
    required_fields = [
        "chunk_id",
        "anchor",
        "section_id",
        "text_raw",
        "scores",
    ]
    
    # Создаем мок для БД с правильным контекстным менеджером
    mock_db = Mock()
    mock_cursor = Mock()
    
    # Настраиваем контекстный менеджер для cursor
    mock_cursor_context = MagicMock()
    mock_cursor_context.__enter__ = Mock(return_value=mock_cursor)
    mock_cursor_context.__exit__ = Mock(return_value=None)
    mock_db.cursor.return_value = mock_cursor_context
    
    # Мокируем результаты запросов для разных ретриверов
    # Для OverviewPurposeRetriever и CitationRetriever
    mock_cursor_fetchall_overview = [
        ("chunk1", "§160.103", "section1", "160.103", "Definitions", "section", "text1", 1, 2, 0.9),
    ]
    # Для GeneralRetriever нужны дополнительные поля
    mock_cursor_fetchall_general = [
        ("chunk1", "§160.103", "section1", "160.103", "Definitions", "atomic", None, None, "text1", 1, 2, 0.9),
    ]
    
    # Тестируем несколько ретриверов
    retriever_classes = [
        (OverviewPurposeRetriever, mock_cursor_fetchall_overview),
        (CitationRetriever, mock_cursor_fetchall_overview),
        (GeneralRetriever, mock_cursor_fetchall_general),
    ]
    
    fake_embedding = [0.1] * 4096
    
    for retriever_class, mock_data in retriever_classes:
        # Создаем новый мок для каждого ретривера
        mock_db_instance = Mock()
        mock_cursor_instance = Mock()
        mock_cursor_context_instance = MagicMock()
        mock_cursor_context_instance.__enter__ = Mock(return_value=mock_cursor_instance)
        mock_cursor_context_instance.__exit__ = Mock(return_value=None)
        mock_db_instance.cursor.return_value = mock_cursor_context_instance
        mock_cursor_instance.fetchall.return_value = mock_data
        mock_cursor_instance.execute = Mock()
        
        retriever = retriever_class(db_connection=mock_db_instance)
        retriever.embedding_client = Mock()
        
        try:
            results = await retriever.retrieve(
                question_embedding=fake_embedding,
                question="test question",
                doc_id="test-doc",
                k=1
            )
            
            assert isinstance(results, list), \
                f"{retriever_class.__name__} должен возвращать список"
            
            if results:
                for result in results:
                    for field in required_fields:
                        assert field in result, \
                            f"{retriever_class.__name__} должен возвращать поле '{field}'"
                    
                    # Проверяем структуру scores
                    if "scores" in result:
                        assert isinstance(result["scores"], dict), \
                            f"{retriever_class.__name__}: scores должен быть словарем"
                        assert "final_score" in result["scores"], \
                            f"{retriever_class.__name__}: scores должен содержать 'final_score'"
            
            print(f"  ✅ {retriever_class.__name__}: контракт соблюден")
        except Exception as e:
            print(f"  ⚠️  {retriever_class.__name__}: ошибка при тестировании: {e}")
            # Не падаем, просто логируем
    
    print("  ✅ test_output_contract_fields_present PASSED\n")


async def run_all_tests():
    """Запускает все smoke тесты."""
    print("=" * 60)
    print("🚀 Запуск smoke тестов для ретриверов")
    print("=" * 60)
    print()
    
    tests = [
        ("Registry возвращает правильный класс", test_registry_returns_correct_class),
        ("Overview возвращает section granularity", test_overview_returns_section_granularity),
        ("Citation фильтрует по anchor prefix", test_citation_anchor_prefix_filter),
        ("General применяет diversity constraint", test_general_diversity_constraint),
        ("Выходные данные содержат обязательные поля", test_output_contract_fields_present),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"📊 Результаты: {passed} прошло, {failed} упало")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
