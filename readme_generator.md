# Answer Generator Architecture

## Обзор

Генератор ответов (`AnswerGenerator`) - это модульная система, которая преобразует найденные ретривером чанки в структурированные ответы с цитатами. Система использует политики генерации (`AnswerPolicy`), которые определяют формат ответа, лимиты контекста и способ обработки.

## Архитектура

```
Question → Classifier → Post-Classification Override → Retriever → Chunks → AnswerGenerator → GenerationResult
                                    ↓                                           ↓
                            Regulatory Principle Detection          AnswerPolicy Selection
                                    ↓                                           ↓
                            Category Override                         Context Building (with limits)
                                                                              ↓
                                                                      Prompt Template Selection
                                                                              ↓
                                                                      LLM Generation (or direct citation)
                                                                              ↓
                                                                      Citation Validation (strict anchor check)
                                                                              ↓
                                                                      GenerationResult
```

### Основные компоненты

1. **Post-Classification Override** - применяет доменные правила после классификации (например, `regulatory_principle` для концепций типа "minimum necessary")
2. **AnswerPolicy** - определяет стратегию генерации ответа
3. **Context Builder** - нормализует и ограничивает контекст по политике
4. **Prompt Templates** - специализированные промпты для каждой политики (с поддержкой `regulatory_principle`)
5. **LLM Client** - обертка над OpenAI-compatible API (VseGPT)
6. **Citation Validator** - валидирует и фильтрует citations из LLM ответов (строгая валидация anchors)

## Маппинг: Category → Retriever → Policy → Output Format

| Category | Retriever | AnswerPolicy | Output Format | Max Context Items | LLM Required |
|----------|-----------|--------------|---------------|-------------------|--------------|
| `citation-required` | `CitationRetriever` | `STRICT_CITATION` | Bullet points: `anchor - text_raw` | 10 | ❌ No (direct) |
| `overview / purpose` | `OverviewPurposeRetriever` | `SUMMARY` | 2-4 sentences, 1 anchor reference | 2 | ✅ Yes |
| `definition` | `DefinitionRetriever` | `QUOTED_ANSWER` | 1-2 sentences + 1-3 exact quotes with anchors (JSON) | 6 | ✅ Yes |
| `regulatory_principle` | `ProceduralRetriever` | `QUOTED_ANSWER` | Special format: "HIPAA does not provide..." + 1-3 exact quotes with anchors (JSON) | 6 | ✅ Yes |
| `procedural / best practices` | `ProceduralRetriever` | `QUOTED_ANSWER` | Yes/No/Unclear + 1-3 exact quotes with anchors (JSON) | 6 | ✅ Yes |
| `scope / applicability` | `ScopeRetriever` | `LISTING` | Structured list, each item with anchor (JSON) | 10 | ✅ Yes |
| `penalties` | `PenaltiesRetriever` | `LISTING` | Structured list, each item with anchor (JSON) | 10 | ✅ Yes |
| `permission / disclosure` | `PermissionDisclosureRetriever` | `LISTING` | Structured list with conditions, each item with anchor (JSON) | 10 | ✅ Yes |
| Navigation questions* | `NavigationRetriever` | `NAVIGATION` | Part/Subpart/Section structure, no text quotes | 10 | ✅ Yes |
| `other` | `GeneralRetriever` | `QUOTED_ANSWER` | Brief answer + 1-3 quotes with anchors (JSON) | 6 | ✅ Yes |

\* Navigation questions определяются по ключевым словам: "which part", "where is", "where are", "where does", "which section", "which subpart"

## Контракты

### Входные данные

#### `AnswerGenerator.generate()`

```python
async def generate(
    question: str,                                    # Вопрос пользователя
    chunks: List[dict],                               # Чанки от ретривера
    classification: QuestionClassification,           # Классификация вопроса
    retriever_signals: Optional[Dict[str, Any]] = None  # Сигналы от ретривера
) -> GenerationResult
```

**Формат `chunks` (вход):**
```python
[
    {
        "chunk_id": str,              # Уникальный ID чанка
        "section_number": str,        # Номер секции (например, "160.103")
        "section_title": str,         # Заголовок секции
        "text_raw": str,              # Дословный текст чанка
        "anchor": Optional[str],      # Anchor для цитирования (например, "§160.103")
        "scores": {                   # Опционально: оценки релевантности
            "final_score": float,
            "vector_score": float,
            "fts_score": float
        },
        "flags": {                    # Опционально: флаги
            "is_seed": bool,
            "is_parent": bool,
            "is_ref": bool,
            "is_sibling": bool
        }
    },
    ...
]
```

**Формат `retriever_signals` (опционально):**
```python
{
    "yesno_signal": Optional[str],      # "yes" | "no" | "unclear" (для procedural)
    "yesno_rationale": Optional[str],   # Объяснение yesno_signal
    "policy_signal": Optional[str]      # "permitted" | "prohibited" | "conditional" | "unclear" (для disclosure)
}
```

### Выходные данные

#### `GenerationResult`

```python
@dataclass
class GenerationResult:
    answer_text: str                  # Текст ответа
    citations: List[Citation]         # Валидированные цитаты
    policy: str                       # Permission policy (для disclosure)
    meta: Dict[str, Any]              # Метаданные для отладки
```

**Формат `Citation`:**
```python
@dataclass
class Citation:
    anchor: str                       # Anchor (например, "§160.103")
    quote: str                        # Точная цитата из text_raw
    chunk_id: Optional[str]           # ID чанка-источника
```

**Формат `meta` (пример):**
```python
{
    "model": str,                     # Использованная модель LLM
    "chunks_count": int,              # Количество входных чанков
    "category": str,                  # Категория вопроса
    "confidence": float,              # Уверенность классификации
    "prompt_template": str,           # Имя использованного шаблона промпта
    "answer_policy": str,             # Использованная политика
    "permission_policy": str,         # Permission policy (если применимо)
    "citations_validated": bool,      # Были ли citations валидированы
    "valid_citations_count": int,     # Количество валидных citations
    "retriever_yesno_signal": Optional[str],      # Сигнал от ретривера
    "retriever_policy_signal": Optional[str]      # Policy signal от ретривера
}
```

## Политики генерации (AnswerPolicy)

### STRICT_CITATION

**Когда используется:**
- Категория: `citation-required`

**Особенности:**
- ❌ **Не использует LLM** - ответ формируется напрямую из контекста
- Формат: bullet points `anchor - text_raw`
- Лимит контекста: 10 элементов
- Все citations валидируются (anchor должен существовать в context)

**Пример:**
```
§164.512(a) - A covered entity may disclose PHI for law enforcement purposes...
§164.512(b) - A covered entity may disclose PHI in response to a law enforcement official's request...
```

### NAVIGATION

**Когда используется:**
- Навигационные вопросы (ключевые слова: "which part", "where is", etc.)
- Категория: может быть любой, но вопрос содержит навигационные ключевые слова

**Особенности:**
- Возвращает структуру документа (Part/Subpart/Section)
- Не цитирует текст, только указывает расположение
- Лимит контекста: 10 элементов

### SUMMARY

**Когда используется:**
- Категория: `overview / purpose`

**Особенности:**
- Краткий обзорный ответ (2-4 предложения)
- Один anchor как reference
- Лимит контекста: 2 элемента (крупные section chunks)

### QUOTED_ANSWER

**Когда используется:**
- Категории: `definition`, `regulatory_principle`, `procedural / best practices`, `other`
- Для `definition`, `regulatory_principle` и `procedural`: citations обязательны

**Особенности:**
- LLM возвращает JSON: `{"answer": "...", "citations": [...]}`
- Citations валидируются (anchor должен быть в context, quote должен быть подстрокой text_raw)
- Для `definition`: если нет валидных citations → "Insufficient context to provide exact citation."
- Для `regulatory_principle`: специальный формат ответа:
  - Начинается с: "HIPAA does not provide a standalone definition of 'X' in the Definitions section."
  - Объясняет нормативный смысл как требование/принцип (не словарное определение)
  - Обязательны 1-3 цитаты с anchors
  - Запрет на придумывание определений вида "X means..."
- Для `procedural`: может включать yesno_signal от ретривера
- Лимит контекста: 6 элементов

### LISTING

**Когда используется:**
- Категории: `scope / applicability`, `penalties`, `permission / disclosure`

**Особенности:**
- LLM возвращает JSON: `{"answer": "...", "citations": [...]}`
- Структурированный список пунктов/условий
- Каждый пункт с anchor
- Для `permission / disclosure`: может включать policy_signal от ретривера
- Лимит контекста: 10 элементов

## Валидация Citations

### Процесс валидации

1. **Парсинг JSON** из LLM ответа (поддержка markdown code fences)
2. **СТРОГАЯ проверка anchor**: anchor должен точно совпадать с anchor в `context_items` (без поблажек, придуманные anchors отклоняются)
3. **Проверка quote**: quote должен быть подстрокой `text_raw` соответствующего context item (после нормализации whitespace)
4. **Автоисправление quote** (если `auto_fix_quote=True`, по умолчанию включено):
   - Если anchor валиден, но quote отсутствует или невалиден → автоматически извлекается релевантный фрагмент из `text_raw`
   - Приоритет извлечения:
     1. Первое предложение (до точки/восклицательного/вопросительного знака)
     2. Первые 300 символов (обрезается по слову)
5. **Фильтрация**: citations с невалидными anchors удаляются (anchor не может быть исправлен автоматически)
6. **Обязательность**: для `definition`, `regulatory_principle` и `procedural` - если после автоисправления нет валидных citations → "Insufficient context to provide exact citation."

### Правила валидации

```python
def validate_citation(
    citation: Dict[str, str], 
    context_items: List[ContextItem],
    auto_fix_quote: bool = True
) -> Optional[Citation]:
    # 1. Проверка наличия anchor
    if not anchor:
        return None
    
    # 2. СТРОГИЙ поиск context_item с таким anchor (точное совпадение)
    matching_item = find_by_anchor(anchor, context_items)  # Строгое сравнение: anchor.strip() == item.anchor.strip()
    if not matching_item:
        # Anchor не найден - citation отклоняется (anchor не может быть исправлен)
        return None
    
    # 3. Если quote отсутствует и auto_fix_quote=True - извлекаем релевантный фрагмент
    if not quote and auto_fix_quote:
        quote = extract_relevant_quote(matching_item.text_raw)
    
    # 4. Проверка, что quote является подстрокой text_raw
    if normalized_quote.lower() not in normalized_text_raw.lower():
        if auto_fix_quote:
            # Автоматически исправляем quote
            quote = extract_relevant_quote(matching_item.text_raw)
        else:
            return None
    
    # 5. Возврат валидной Citation
    return Citation(anchor=anchor, quote=quote, chunk_id=matching_item.chunk_id)
```

### Автоисправление quote

**Когда используется:**
- Anchor валиден (существует в context_items)
- Quote отсутствует или не является подстрокой text_raw
- `auto_fix_quote=True` (по умолчанию)

**Алгоритм извлечения релевантного фрагмента:**
1. Пытается найти первое предложение (до `.`, `!`, `?`)
2. Если предложение не слишком длинное (≤ 300 символов) → использует его
3. Иначе берет первые 300 символов, обрезая по последнему пробелу

**Преимущества:**
- Улучшает UX для definition вопросов (даже если LLM вернул невалидный quote, citation сохраняется)
- Сохраняет полезные citations с валидными anchors
- Снижает количество случаев "Insufficient context" для definition

## Примеры

### Пример 1: Citation-Required (STRICT_CITATION)

**Вход:**
```python
question = "Cite the specific regulation texts regarding permitted disclosures to law enforcement."
classification = QuestionClassification(
    category="citation-required",
    confidence=0.95,
    reasoning="Explicit request for citations"
)
chunks = [
    {
        "chunk_id": "chunk_164_512_a",
        "section_number": "164.512",
        "section_title": "Disclosures for law enforcement purposes",
        "text_raw": "A covered entity may disclose PHI for law enforcement purposes as required by law...",
        "anchor": "§164.512(a)"
    },
    {
        "chunk_id": "chunk_164_512_b",
        "section_number": "164.512",
        "section_title": "Disclosures for law enforcement purposes",
        "text_raw": "A covered entity may disclose PHI in response to a law enforcement official's request...",
        "anchor": "§164.512(b)"
    }
]
```

**Процесс:**
1. Policy: `STRICT_CITATION` (выбрана по категории)
2. Context: 2 элемента (в пределах лимита 10)
3. **LLM пропущен** - ответ формируется напрямую
4. Ответ собирается как bullet points

**Выход:**
```python
GenerationResult(
    answer_text="§164.512(a) - A covered entity may disclose PHI for law enforcement purposes as required by law...\n§164.512(b) - A covered entity may disclose PHI in response to a law enforcement official's request...",
    citations=[
        Citation(anchor="§164.512(a)", quote="A covered entity may disclose PHI for law enforcement purposes as required by law...", chunk_id="chunk_164_512_a"),
        Citation(anchor="§164.512(b)", quote="A covered entity may disclose PHI in response to a law enforcement official's request...", chunk_id="chunk_164_512_b")
    ],
    policy="",
    meta={
        "policy": "strict_citation",
        "llm_skipped": True,
        "citations_count": 2,
        "context_items_count": 2,
        "answer_policy": "strict_citation",
        ...
    }
)
```

### Пример 2: Definition (QUOTED_ANSWER)

**Вход:**
```python
question = "What does business associate mean?"
classification = QuestionClassification(
    category="definition",
    confidence=0.92,
    reasoning="Question about term definition"
)
chunks = [
    {
        "chunk_id": "chunk_160_103",
        "section_number": "160.103",
        "section_title": "Definitions",
        "text_raw": "Business associate means a person or entity that performs certain functions or activities on behalf of a covered entity...",
        "anchor": "§160.103"
    }
]
```

**Процесс:**
1. Policy: `QUOTED_ANSWER` (выбрана по категории `definition`)
2. Context: 1 элемент (в пределах лимита 6)
3. Prompt: `build_quoted_answer_prompt()` с инструкцией возвращать JSON
4. LLM возвращает JSON:
   ```json
   {
     "answer": "Business associate means a person or entity that performs certain functions or activities on behalf of a covered entity.",
     "citations": [
       {"anchor": "§160.103", "quote": "Business associate means a person or entity"}
     ]
   }
   ```
5. Валидация citations:
   - Anchor "§160.103" найден в context ✓
   - Quote "Business associate means a person or entity" является подстрокой text_raw ✓
   - Citation валидна ✓

**Выход:**
```python
GenerationResult(
    answer_text="Business associate means a person or entity that performs certain functions or activities on behalf of a covered entity.",
    citations=[
        Citation(anchor="§160.103", quote="Business associate means a person or entity", chunk_id="chunk_160_103")
    ],
    policy="",
    meta={
        "model": "anthropic/claude-3-haiku",
        "chunks_count": 1,
        "category": "definition",
        "confidence": 0.92,
        "prompt_template": "build_quoted_answer_prompt",
        "answer_policy": "quoted_answer",
        "citations_validated": True,
        "valid_citations_count": 1,
        ...
    }
)
```

**Сценарий с невалидными citations:**

**Случай 1: Невалидный anchor**
Если LLM вернул:
```json
{
  "answer": "Business associate means...",
  "citations": [
    {"anchor": "§999.999", "quote": "Invalid anchor"}
  ]
}
```

Валидация:
- Anchor "§999.999" не найден в context ✗
- Citation отклонена (anchor не может быть исправлен)

Результат:
```python
GenerationResult(
    answer_text="Insufficient context to provide exact citation.",
    citations=[],
    policy="",
    meta={...}
)
```

**Случай 2: Валидный anchor, невалидный quote (автоисправление)**
Если LLM вернул:
```json
{
  "answer": "Business associate means a person or entity...",
  "citations": [
    {"anchor": "§160.103", "quote": "This quote does not match the text"}
  ]
}
```

Валидация:
- Anchor "§160.103" найден в context ✓
- Quote не найден в text_raw ✗
- **Автоисправление**: quote заменяется на релевантный фрагмент из text_raw
- Citation становится валидной ✓

Результат:
```python
GenerationResult(
    answer_text="Business associate means a person or entity...",
    citations=[
        Citation(
            anchor="§160.103", 
            quote="Business associate means a person or entity that performs certain functions.",  # Автоисправлено
            chunk_id="chunk_160_103"
        )
    ],
    policy="",
    meta={
        ...
        "auto_fixed_citations_count": 1  # В логах
    }
)
```

**Случай 3: Валидный anchor, отсутствующий quote (автоисправление)**
Если LLM вернул:
```json
{
  "answer": "Business associate means...",
  "citations": [
    {"anchor": "§160.103"}
  ]
}
```

Валидация:
- Anchor "§160.103" найден в context ✓
- Quote отсутствует
- **Автоисправление**: quote извлекается из text_raw
- Citation становится валидной ✓

## Сигналы от ретриверов

### Yes/No Signal (Procedural)

Ретривер `ProceduralRetriever` может вернуть сигнал:
```python
retriever_signals = {
    "yesno_signal": "yes",  # или "no", "unclear"
    "yesno_rationale": "Found explicit term 'encryption' in §164.312..."
}
```

Генератор передает этот сигнал в промпт:
```
Retriever signal: YES (Retriever signal: Found explicit term 'encryption' in §164.312...). 
Start your answer with: YES / NO / UNCLEAR based on the evidence in the provided context 
(the signal is a hint, verify with context).
```

### Policy Signal (Disclosure)

Ретривер `PermissionDisclosureRetriever` может вернуть сигнал:
```python
retriever_signals = {
    "policy_signal": "conditional"  # или "permitted", "prohibited", "unclear"
}
```

Генератор передает этот сигнал в промпт:
```
Retriever policy signal: CONDITIONAL (permitted/conditional/prohibited/unclear). 
This is a hint - verify with the provided context and reflect conditions/limitations in your answer.
```

## Лимиты контекста по политике

| Policy | Max Items | Причина |
|--------|-----------|---------|
| `STRICT_CITATION` | 10 | Нужно собрать несколько цитат |
| `SUMMARY` | 2 | Обзорный ответ, нужны крупные chunks |
| `LISTING` | 10 | Перечисления могут быть длинными |
| `QUOTED_ANSWER` | 6 | Баланс между точностью и контекстом |
| `NAVIGATION` | 10 | Навигация может требовать обзора структуры |

## Использование в app.py

```python
from generator import get_generator

generator = get_generator()

# Извлечение сигналов от ретривера
retriever_signals = {}
if classification.category == "procedural / best practices" and retrieved_chunks_raw:
    first_chunk = retrieved_chunks_raw[0]
    if "yesno_signal" in first_chunk:
        retriever_signals["yesno_signal"] = first_chunk.get("yesno_signal")
        retriever_signals["yesno_rationale"] = first_chunk.get("yesno_rationale", "")

# Генерация ответа
generation_result = await generator.generate(
    question=request.question,
    chunks=chunks_for_generator,
    classification=classification,
    retriever_signals=retriever_signals
)

# Формирование sources из citations
sources = []
if generation_result.citations:
    for citation in generation_result.citations:
        if citation.anchor:
            sources.append(citation.anchor)
        elif citation.chunk_id:
            sources.append(citation.chunk_id)
```

## Модульная структура

```
generator/
├── __init__.py              # Публичные экспорты
├── base.py                  # GenerationResult, Citation, ContextItem
├── policy.py                # AnswerPolicy, PermissionPolicy, choose_policy()
├── context_builder.py       # build_context() - нормализация и лимиты
├── llm_client.py           # LLMClient - обертка над OpenAI API
├── citation_validator.py   # Валидация citations из LLM
├── generator.py            # AnswerGenerator - основная оркестрация
└── prompts/
    ├── __init__.py
    ├── strict_citation.py  # Placeholder (не используется)
    ├── navigation.py        # build_navigation_prompt()
    ├── summary.py          # build_summary_prompt()
    ├── quoted_answer.py    # build_quoted_answer_prompt()
    └── listing.py          # build_listing_prompt()
```

## Тестирование

Smoke тесты находятся в `test_generator_smoke.py`:
- `test_policy_selection_by_category` - выбор политики
- `test_strict_citation_skips_llm` - STRICT_CITATION обходит LLM
- `test_definition_requires_citation_or_insufficient` - обязательность citations
- `test_citation_validation_rejects_unknown_anchor` - валидация citations
- `test_context_limits_by_policy` - лимиты контекста

Запуск:
```bash
uv run python test_generator_smoke.py
```
