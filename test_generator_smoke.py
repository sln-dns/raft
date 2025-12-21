"""Smoke тесты для проверки контрактов генератора.

Эти тесты не зависят от LLM API и БД, проверяют только контракты и базовую логику.

Запуск:
    python test_generator_smoke.py
    # или
    python -m test_generator_smoke

Тесты:
    - test_policy_selection_by_category: проверка выбора политики по категории
    - test_strict_citation_skips_llm: проверка, что STRICT_CITATION не вызывает LLM
    - test_definition_requires_citation_or_insufficient: проверка обязательности citations для definition
    - test_citation_validation_rejects_unknown_anchor: проверка валидации citations
    - test_context_limits_by_policy: проверка лимитов контекста по политике
"""

import asyncio
import sys
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Мокируем зависимости до импорта модулей
sys.modules['openai'] = MagicMock()
mock_os = MagicMock()
mock_os.getenv = MagicMock(return_value="test-key")
sys.modules['os'] = mock_os
sys.modules['dotenv'] = MagicMock()
sys.modules['classifier'] = MagicMock()

# Импортируем модули после мокирования зависимостей
from generator.policy import AnswerPolicy, choose_policy
from generator.generator import AnswerGenerator
from generator.base import ContextItem, Citation
from generator.context_builder import build_context
from generator.citation_validator import validate_citation, parse_and_validate_citations

# Создаем мок для QuestionClassification
class QuestionClassification:
    def __init__(self, category: str, confidence: float, reasoning: str = ""):
        self.category = category
        self.confidence = confidence
        self.reasoning = reasoning


def test_policy_selection_by_category():
    """Тест: выбор политики по категории вопроса."""
    print("🧪 test_policy_selection_by_category...")
    
    test_cases = [
        ("citation-required", AnswerPolicy.STRICT_CITATION),
        ("overview / purpose", AnswerPolicy.SUMMARY),
        ("definition", AnswerPolicy.QUOTED_ANSWER),
        ("regulatory_principle", AnswerPolicy.QUOTED_ANSWER),
        ("procedural / best practices", AnswerPolicy.QUOTED_ANSWER),
        ("scope / applicability", AnswerPolicy.LISTING),
        ("penalties", AnswerPolicy.LISTING),
        ("permission / disclosure", AnswerPolicy.LISTING),
        ("other", AnswerPolicy.QUOTED_ANSWER),
    ]
    
    for category, expected_policy in test_cases:
        policy = choose_policy(
            category=category,
            classification_confidence=0.9,
            signals={},
            question=None
        )
        assert policy == expected_policy, f"Category '{category}' should return {expected_policy}, got {policy}"
    
    # Проверка навигационных вопросов
    nav_policy = choose_policy(
        category="other",
        classification_confidence=0.9,
        signals={},
        question="Which part covers privacy?"
    )
    assert nav_policy == AnswerPolicy.NAVIGATION, f"Navigation question should return NAVIGATION, got {nav_policy}"
    
    print("✅ test_policy_selection_by_category passed")


async def test_strict_citation_skips_llm():
    """Тест: STRICT_CITATION политика не вызывает LLM."""
    print("🧪 test_strict_citation_skips_llm...")
    
    # Мокаем LLMClient
    mock_llm_client = AsyncMock()
    mock_llm_client.complete = AsyncMock()
    mock_llm_client.model = "test-model"
    
    # Создаем тестовые данные
    chunks = [
        {
            "chunk_id": "chunk1",
            "section_number": "164.512",
            "section_title": "Disclosures for law enforcement",
            "text_raw": "A covered entity may disclose PHI for law enforcement purposes...",
            "anchor": "§164.512(a)"
        }
    ]
    
    classification = QuestionClassification(
        category="citation-required",
        confidence=0.9,
        reasoning="Test"
    )
    
    # Создаем генератор с мокнутым LLMClient
    # Используем patch для обхода инициализации LLMClient
    with patch('generator.generator.LLMClient') as mock_llm_class:
        mock_llm_class.return_value = mock_llm_client
        generator = AnswerGenerator(api_key="test-key", base_url="http://test", model="test-model")
        # Убеждаемся, что используется наш мок
        generator.llm_client = mock_llm_client
        
        # Вызываем генератор
        result = await generator.generate(
            question="Cite the specific regulation texts regarding permitted disclosures to law enforcement.",
            chunks=chunks,
            classification=classification,
            retriever_signals={}
        )
        
        # Проверяем, что LLM не был вызван
        mock_llm_client.complete.assert_not_called()
        
        # Проверяем результат
        assert result.answer_text.startswith("§164.512(a)"), "Answer should start with anchor"
        assert "§164.512(a)" in result.answer_text, "Answer should contain anchor"
        assert len(result.citations) > 0, "Should have citations"
        assert result.meta.get("llm_skipped") == True, "Should have llm_skipped flag"
    
    print("✅ test_strict_citation_skips_llm passed")


async def test_definition_requires_citation_or_insufficient():
    """Тест: definition требует citation или возвращает 'Insufficient context'."""
    print("🧪 test_definition_requires_citation_or_insufficient...")
    
    # Мокаем LLMClient для возврата JSON без валидных citations
    mock_llm_client = AsyncMock()
    
    # Сценарий 1: LLM возвращает JSON с невалидными citations (unknown anchor)
    mock_llm_client.complete = AsyncMock(return_value='''{
        "answer": "Business associate means a person or entity...",
        "citations": [
            {"anchor": "§999.999", "quote": "Invalid anchor"}
        ]
    }''')
    mock_llm_client.model = "test-model"
    
    chunks = [
        {
            "chunk_id": "chunk1",
            "section_number": "160.103",
            "section_title": "Definitions",
            "text_raw": "Business associate means a person or entity that performs certain functions...",
            "anchor": "§160.103"
        }
    ]
    
    classification = QuestionClassification(
        category="definition",
        confidence=0.9,
        reasoning="Test"
    )
    
    # Создаем генератор с мокнутым LLMClient
    with patch('generator.generator.LLMClient') as mock_llm_class:
        mock_llm_class.return_value = mock_llm_client
        generator = AnswerGenerator(api_key="test-key", base_url="http://test", model="test-model")
        generator.llm_client = mock_llm_client
        
        # Вызываем генератор
        result = await generator.generate(
            question="What does business associate mean?",
            chunks=chunks,
            classification=classification,
            retriever_signals={}
        )
        
        # Для definition require_citations=True, поэтому если нет валидных citations - должен вернуть "Insufficient context"
        # Но в данном случае anchor не найден в context, поэтому citations будут пустыми
        # И должен вернуться "Insufficient context to provide exact citation."
        assert "Insufficient context" in result.answer_text, "Should return 'Insufficient context' when no valid citations"
        assert len(result.citations) == 0, "Should have no valid citations"
        
        # Сценарий 2: LLM возвращает JSON с валидными citations
        mock_llm_client.complete = AsyncMock(return_value='''{
            "answer": "Business associate means a person or entity...",
            "citations": [
                {"anchor": "§160.103", "quote": "Business associate means a person or entity"}
            ]
        }''')
        
        # Используем тот же генератор
        result2 = await generator.generate(
            question="What does business associate mean?",
            chunks=chunks,
            classification=classification,
            retriever_signals={}
        )
    
        # Должен быть валидный ответ с citations
        assert "Insufficient context" not in result2.answer_text, "Should not return 'Insufficient context' when valid citations exist"
        assert len(result2.citations) > 0, "Should have valid citations"
        assert result2.citations[0].anchor == "§160.103", "Citation should have correct anchor"
    
    print("✅ test_definition_requires_citation_or_insufficient passed")


def test_citation_validation_rejects_unknown_anchor():
    """Тест: валидация отклоняет citations с неизвестными anchors."""
    print("🧪 test_citation_validation_rejects_unknown_anchor...")
    
    # Создаем контекст с известными anchors
    context_items = [
        ContextItem(
            chunk_id="chunk1",
            section_number="160.103",
            section_title="Definitions",
            text_raw="Business associate means a person or entity that performs certain functions or activities on behalf of a covered entity.",
            anchor="§160.103"
        ),
        ContextItem(
            chunk_id="chunk2",
            section_number="164.502",
            section_title="Uses and disclosures",
            text_raw="A covered entity may use or disclose PHI...",
            anchor="§164.502"
        )
    ]
    
    # Тест 1: Валидная citation (anchor существует в context)
    valid_citation = {"anchor": "§160.103", "quote": "Business associate means"}
    validated = validate_citation(valid_citation, context_items)
    assert validated is not None, "Valid citation should pass validation"
    assert validated.anchor == "§160.103", "Validated citation should have correct anchor"
    
    # Тест 2: Невалидная citation (anchor не существует в context)
    invalid_citation = {"anchor": "§999.999", "quote": "Some text"}
    validated_invalid = validate_citation(invalid_citation, context_items, auto_fix_quote=False)
    assert validated_invalid is None, "Invalid citation (unknown anchor) should be rejected"
    
    # Тест 3: Citation с quote, который не является подстрокой text_raw (без автоисправления)
    invalid_quote_citation = {"anchor": "§160.103", "quote": "This quote does not exist in the text"}
    validated_quote = validate_citation(invalid_quote_citation, context_items, auto_fix_quote=False)
    assert validated_quote is None, "Citation with quote not in text_raw should be rejected when auto_fix_quote=False"
    
    # Тест 4: Citation с невалидным quote, но валидным anchor (с автоисправлением)
    invalid_quote_auto_fix = {"anchor": "§160.103", "quote": "This quote does not exist in the text"}
    validated_auto_fixed = validate_citation(invalid_quote_auto_fix, context_items, auto_fix_quote=True)
    assert validated_auto_fixed is not None, "Citation with valid anchor should be auto-fixed when auto_fix_quote=True"
    assert validated_auto_fixed.anchor == "§160.103", "Auto-fixed citation should have correct anchor"
    assert "Business associate means" in validated_auto_fixed.quote, "Auto-fixed quote should contain relevant text"
    
    # Тест 5: Citation без quote, но с валидным anchor (с автоисправлением)
    no_quote_auto_fix = {"anchor": "§160.103"}
    validated_no_quote = validate_citation(no_quote_auto_fix, context_items, auto_fix_quote=True)
    assert validated_no_quote is not None, "Citation with valid anchor but no quote should be auto-fixed"
    assert len(validated_no_quote.quote) > 0, "Auto-fixed citation should have quote"
    
    # Тест 6: Citation без anchor
    no_anchor_citation = {"quote": "Some text"}
    validated_no_anchor = validate_citation(no_anchor_citation, context_items)
    assert validated_no_anchor is None, "Citation without anchor should be rejected"
    
    print("✅ test_citation_validation_rejects_unknown_anchor passed")


def test_context_limits_by_policy():
    """Тест: build_context ограничивает количество элементов по политике."""
    print("🧪 test_context_limits_by_policy...")
    
    # Создаем много чанков (больше лимита для любой политики)
    chunks = [
        {
            "chunk_id": f"chunk{i}",
            "section_number": f"160.{i:03d}",
            "section_title": f"Section {i}",
            "text_raw": f"Text {i}",
            "anchor": f"§160.{i:03d}"
        }
        for i in range(20)  # 20 чанков
    ]
    
    # Тест для каждой политики
    test_cases = [
        (AnswerPolicy.STRICT_CITATION, 10),  # Лимит 10
        (AnswerPolicy.SUMMARY, 2),  # Лимит 2
        (AnswerPolicy.LISTING, 10),  # Лимит 10
        (AnswerPolicy.QUOTED_ANSWER, 6),  # Лимит 6
        (AnswerPolicy.NAVIGATION, 10),  # Лимит 10
    ]
    
    for policy, expected_limit in test_cases:
        context_items = build_context(chunks, policy)
        assert len(context_items) <= expected_limit, (
            f"Policy {policy.value} should limit context to {expected_limit}, "
            f"got {len(context_items)}"
        )
        # Проверяем, что контекст отсортирован
        if len(context_items) > 1:
            # Проверяем сортировку по section_number
            section_numbers = [item.section_number for item in context_items]
            assert section_numbers == sorted(section_numbers), "Context should be sorted by section_number"
    
    # Тест для пустого списка
    empty_context = build_context([], AnswerPolicy.QUOTED_ANSWER)
    assert len(empty_context) == 0, "Empty chunks should return empty context"
    
    print("✅ test_context_limits_by_policy passed")


async def run_all_tests():
    """Запускает все тесты."""
    print("=" * 60)
    print("Запуск smoke тестов для генератора")
    print("=" * 60)
    
    tests = [
        ("test_policy_selection_by_category", test_policy_selection_by_category),
        ("test_strict_citation_skips_llm", test_strict_citation_skips_llm),
        ("test_definition_requires_citation_or_insufficient", test_definition_requires_citation_or_insufficient),
        ("test_citation_validation_rejects_unknown_anchor", test_citation_validation_rejects_unknown_anchor),
        ("test_context_limits_by_policy", test_context_limits_by_policy),
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
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Результаты: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
