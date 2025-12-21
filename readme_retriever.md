# Документация по ретриверам

## Обзор

Система использует модульную архитектуру ретриверов для поиска релевантных чанков в зависимости от категории вопроса. Каждый ретривер специализируется на определенном типе вопросов и использует специфические алгоритмы поиска.

## Базовый интерфейс

Все ретриверы наследуются от `BaseRetriever` и реализуют метод `retrieve()`:

```python
async def retrieve(
    self,
    question_embedding: List[float],  # Эмбеддинг вопроса (размерность 4096)
    max_results: int = 5,              # Максимальное количество результатов
    question: Optional[str] = None,    # Текст вопроса
    **kwargs                           # Дополнительные параметры
) -> List[Dict[str, Any]]:             # Список словарей с информацией о чанках
```

## Автоматический выбор ретривера

Ретривер выбирается автоматически на основе категории вопроса через функцию `get_retriever_for_category()`:

```python
retriever = get_retriever_for_category(
    category="overview / purpose",  # Категория из классификатора
    question="What is the purpose of HIPAA?",  # Текст вопроса (для навигации)
    db_connection=None  # Опционально: существующее подключение к БД
)
```

### Маппинг категорий

| Категория | Ретривер | Примечание |
|-----------|----------|------------|
| `overview / purpose` | `OverviewPurposeRetriever` | Общий обзор и назначение |
| `definition` | `DefinitionRetriever` | Определения терминов |
| `regulatory_principle` | `ProceduralRetriever` | Регуляторные принципы (minimum necessary, reasonable safeguards и т.д.) |
| `scope / applicability` | `ScopeRetriever` | Область применения |
| `penalties` | `PenaltiesRetriever` | Штрафы и санкции |
| `procedural / best practices` | `ProceduralRetriever` | Процедуры и best practices |
| `permission / disclosure` | `PermissionDisclosureRetriever` | Разрешения на раскрытие |
| `citation-required` | `CitationRetriever` | Строгое цитирование |
| `other` | `GeneralRetriever` | Fallback для прочих вопросов |
| *навигационные вопросы* | `NavigationRetriever` | Автоматически по ключевым словам |

**Навигационные вопросы** определяются автоматически по ключевым словам: "which part", "where is", "where are", "where does", "which section", "which subpart".

---

## 1. OverviewPurposeRetriever

### Назначение
Для вопросов об общем обзоре и назначении регуляций. Возвращает 1-2 крупных section-level фрагмента, описывающих общий смысл/назначение.

### Примеры вопросов
- "What is the overall purpose of HIPAA regulations?"
- "What is the purpose of Part 160?"
- "What does HIPAA cover?"

### Входные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `question_embedding` | `List[float]` | - | Эмбеддинг вопроса (4096 dim) |
| `max_results` | `int` | `2` | Максимальное количество результатов |
| `question` | `Optional[str]` | `None` | Текст вопроса (для FTS) |
| `part` | `Optional[int]` | `None` | Номер части (например, 160) |
| `doc_id` | `Optional[str]` | `"hipaa-reg-2013-03-26"` | ID документа |

### Выходные данные

```python
[
    {
        "chunk_id": "chk:160.102:section",
        "anchor": "§160.102",
        "section_id": "sec:160.102",
        "section_number": "160.102",
        "section_title": "Applicability",
        "text_raw": "...",
        "page_start": 1,
        "page_end": 2,
        "scores": {
            "vector_score": 0.5526,
            "fts_score": 0.8000,
            "final_score": 0.6268
        }
    },
    ...
]
```

### Алгоритм
1. **FTS поиск** по `section_title` с ключевыми словами (purpose, basis, scope)
2. **Vector similarity поиск** по section-level chunks
3. **Объединение и ранжирование**: `final_score = 0.7 * vector_score + 0.3 * fts_score_norm`
4. **Дедупликация** по `section_id`
5. **Выбор top-k** (по умолчанию 2)

### Фильтры
- `granularity = 'section'`
- `part` (опционально)
- `embedding IS NOT NULL`

---

## 2. DefinitionRetriever

### Назначение
Для вопросов о терминах. Возвращает точное определение из нормативного текста (1-2 атомарных подпараграфа) с точной цитатой и anchor.

### Примеры вопросов
- "What does 'minimum necessary' mean in HIPAA terminology?"
- "Define 'business associate'."
- "What is a covered entity under HIPAA?"

### Входные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `question_embedding` | `List[float]` | - | Эмбеддинг вопроса (4096 dim) |
| `max_results` | `int` | `1` | Максимальное количество результатов (макс. 2) |
| `question` | `Optional[str]` | `None` | Текст вопроса |
| `doc_id` | `Optional[str]` | `"hipaa-reg-2013-03-26"` | ID документа |
| `k` | `int` | `1` | Количество результатов (1-2) |
| `strict_quote` | `bool` | `True` | Требовать точную цитату |

### Выходные данные

```python
[
    {
        "chunk_id": "chk:160.103:Business_associate(1)",
        "anchor": "§160.103:Business_associate(1)",
        "section_id": "sec:160.103",
        "section_number": "160.103",
        "section_title": "Definitions",
        "text_raw": "Business associate: (1) Except as provided...",
        "page_start": 10,
        "page_end": 11,
        "scores": {
            "def_table_score": 0.0,
            "vector_score": 0.8390,
            "fts_score": 0.5000,
            "final_score": 0.8390
        },
        "term": "business associate",
        "explanation": "fallback semantic search (term: business associate)"
    }
]
```

### Алгоритм
1. **Извлечение термина** из вопроса (кавычки, шаблоны "what does X mean", "define X")
2. **Поиск в таблице definitions** (если есть) - exact match и fuzzy match
3. **Fallback поиск** по atomic chunks:
   - Фильтры: `granularity = 'atomic'`, `chunk_kind = 'definition'`
   - FTS по term (если есть) или по question
   - Гибридный scoring: `0.6*vector + 0.4*fts` (если term есть) или `0.8*vector + 0.2*fts`
4. **Валидация результатов**: проверка паттернов определения ("means", "refers to"), секции Definitions, наличия anchor

### Фильтры
- `granularity = 'atomic'`
- `chunk_kind = 'definition'`

---

## 3. ScopeRetriever

### Назначение
Для вопросов о применимости. Возвращает набор релевантных подпунктов (обычно 2-6 atomic чанков) с сохранением anchor'ов для цитирования.

### Примеры вопросов
- "Which entities are specifically regulated under HIPAA?"
- "To whom does this subchapter apply?"
- "Which entities are covered entities / business associates?"

### Входные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `question_embedding` | `List[float]` | - | Эмбеддинг вопроса (4096 dim) |
| `max_results` | `int` | `6` | Максимальное количество результатов |
| `question` | `Optional[str]` | `None` | Текст вопроса |
| `doc_id` | `Optional[str]` | `"hipaa-reg-2013-03-26"` | ID документа |
| `part` | `Optional[int]` | `None` | Номер части (например, 160 или 164) |
| `k` | `int` | `6` | Количество итоговых пунктов |
| `seed_k` | `int` | `3` | Количество seed-чанков перед расширением |
| `expand_with_siblings` | `bool` | `True` | Расширять соседями |

### Выходные данные

```python
[
    {
        "chunk_id": "chk:160.102:(a)",
        "anchor": "§160.102(a)",
        "section_id": "sec:160.102",
        "section_number": "160.102",
        "section_title": "Applicability",
        "paragraph_path": "(a)",
        "text_raw": "(a) Except as otherwise provided...",
        "page_start": 1,
        "page_end": 2,
        "scores": {
            "vector_score": 0.4661,
            "fts_score": 0.5000,
            "final_score": 0.4661
        },
        "explanation": "seed from semantic search"
    },
    ...
]
```

### Алгоритм
1. **Подготовка FTS query** с усилением (apply, applicability, entity, entities, covered, regulated)
2. **Seed retrieval** (hybrid):
   - Фильтры: `granularity = 'atomic'`, опционально `part`
   - Scoring: `final_score = 0.7*vector_score + 0.3*fts_score_norm`
   - Выбор `seed_k=3` лучших с дедупом по `section_id`
3. **Sibling expansion**:
   - По `parent_chunk_id` (предпочтительно)
   - Fallback по `section_id`
   - Scoring siblings: `final_score = seed.final_score * 0.95`
4. **Дедупликация и сортировка** по `section_id`, затем по `anchor`

### Фильтры
- `granularity = 'atomic'`
- `part` (опционально)

---

## 4. PenaltiesRetriever

### Назначение
Для вопросов о санкциях/штрафах. Возвращает несколько релевантных подпунктов (обычно 3-8 atomic чанков) с суммами/диапазонами штрафов, категориями нарушений, условиями применения санкций.

### Примеры вопросов
- "What are the potential civil penalties for noncompliance?"
- "What are the penalties for HIPAA violations?"
- "What is the maximum civil monetary penalty?"

### Входные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `question_embedding` | `List[float]` | - | Эмбеддинг вопроса (4096 dim) |
| `max_results` | `int` | `6` | Максимальное количество результатов |
| `question` | `Optional[str]` | `None` | Текст вопроса |
| `doc_id` | `Optional[str]` | `"hipaa-reg-2013-03-26"` | ID документа |
| `parts` | `Optional[List[int]]` | `[160, 164]` | Список частей для поиска |
| `k` | `int` | `6` | Количество итоговых фрагментов |
| `seed_k` | `int` | `4` | Количество seed-чанков |
| `expand_section` | `bool` | `True` | Добавлять соседние подпункты в той же секции |
| `need_amounts` | `bool` | `True` | Усиливать наличие чисел/денежных сумм |

### Выходные данные

```python
[
    {
        "chunk_id": "chk:160.404:(b)",
        "anchor": "§160.404(b)",
        "section_id": "sec:160.404",
        "section_number": "160.404",
        "section_title": "Amount of a civil money penalty",
        "paragraph_path": "(b)",
        "text_raw": "(b) The amount of a civil money penalty...",
        "page_start": 50,
        "page_end": 51,
        "scores": {
            "vector_score": 0.7206,
            "fts_score": 0.5000,
            "amount_score": 1.0000,
            "final_score": 0.7206
        },
        "explanation": "penalty filter + hybrid retrieval"
    },
    ...
]
```

### Алгоритм
1. **Подготовка FTS query** с усилением (penalty, penalties, civil, fine, violation, amount, maximum, tier, dollar)
2. **Candidate retrieval** (hybrid):
   - Фильтры: `granularity = 'atomic'`, `part IN (160, 164)`, `chunk_kind IN ('scope', 'requirement', 'other')`
   - Scoring: `base_score = 0.65*vector_score + 0.35*fts_score_norm`
3. **Amount scoring**: усиление чанков с числами/суммами ($, USD, числа с разделителями, ключевые слова)
   - `final_score = 0.55*base_score + 0.45*amount_score` (если `need_amounts=True`)
4. **Seed selection**: выбор `seed_k=4` лучших с дедупом по `section_id`
5. **Expansion**: расширение соседями по `section_id` (только с `amount_score>0` если `need_amounts=True`)
6. **Дедупликация и сортировка** по `section_id`, затем по `anchor`

### Фильтры
- `granularity = 'atomic'`
- `part IN (160, 164)`
- `chunk_kind IN ('scope', 'requirement', 'other')` (fallback, так как 'penalty' нет в схеме)

---

## 5. ProceduralRetriever

### Назначение
Для вопросов о процедурах/best practices. Отвечает на вопросы вида "mentions/does it require/recommend best practices" для тем безопасности/процедур. Возвращает 2-6 релевантных atomic чанков с anchor и формирует сигнал yes/no для answer-слоя.

### Примеры вопросов
- "Does HIPAA mention encryption best practices?"
- "Does HIPAA require encryption?"
- "What safeguards are required for electronic PHI?"

### Входные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `question_embedding` | `List[float]` | - | Эмбеддинг вопроса (4096 dim) |
| `max_results` | `int` | `4` | Максимальное количество результатов |
| `question` | `Optional[str]` | `None` | Текст вопроса |
| `doc_id` | `Optional[str]` | `"hipaa-reg-2013-03-26"` | ID документа |
| `parts` | `Optional[List[int]]` | `[164]` | Список частей (по умолчанию Part 164) |
| `k` | `int` | `4` | Количество итоговых пунктов |
| `seed_k` | `int` | `3` | Количество seed-чанков |
| `expand_section` | `bool` | `True` | Добавлять соседние подпункты в той же секции |
| `need_yesno` | `bool` | `True` | Формировать сигнал yes/no (только для `procedural / best practices`) |
| `category` | `Optional[str]` | `None` | Категория вопроса (используется для `regulatory_principle`) |

### Выходные данные

```python
[
    {
        "chunk_id": "chk:164.306:(a)",
        "anchor": "§164.306(a)",
        "section_id": "sec:164.306",
        "section_number": "164.306",
        "section_title": "Security standards: General rules",
        "paragraph_path": "(a)",
        "text_raw": "(a) General requirements...",
        "page_start": 100,
        "page_end": 101,
        "scores": {
            "vector_score": 0.5046,
            "fts_score": 0.5000,
            "keyword_score": 0.8000,
            "final_score": 0.5046
        },
        "explanation": "procedural retrieval",
        "yesno_signal": "unclear",  # "yes" | "no" | "unclear"
        "yesno_rationale": "Found safeguards/implementation specifications in §164.306(a), but no explicit encryption mention"
    },
    ...
]
```

### Алгоритм
1. **Определение под-темы** и ключевых токенов:
   - Для обычных procedural вопросов: encryption, security, safeguards
   - Для `regulatory_principle` (если `category="regulatory_principle"`):
     - Определяет тему (например, "minimum_necessary")
     - Добавляет topic tokens: "minimum necessary", "reasonable efforts", "limit", "accomplish the intended purpose", "minimum amount", "reasonably necessary"
2. **Подготовка FTS query** с усилением (security, safeguard, technical, administrative, physical, implementation specification + topic tokens)
3. **Candidate retrieval** (hybrid):
   - Фильтры: `granularity = 'atomic'`, опционально `part IN [164]`
   - Fallback: если по Part 164 мало кандидатов, снимает фильтр part
4. **Keyword evidence scoring**: усиление чанков с ключевыми словами
   - +1.0 если есть encryption/encrypt (для encryption-темы)
   - +1.0 если есть minimum necessary tokens (для regulatory_principle темы "minimum_necessary")
   - +0.6 если есть implementation specification или addressable
   - +0.4 если есть safeguard(s) или security
   - +0.4 если есть модальные "must/required/shall"
   - -0.3 если встречается "not required"
5. **Soft boost для regulatory principles** (если `category="regulatory_principle"`):
   - Для темы "minimum_necessary": применяет boost (×1.1) к final_score для anchors, начинающихся с "§164.502" или "§164.514"
6. **Merge + final scoring**: `final_score = 0.55*vector_score + 0.20*fts_score_norm + 0.25*keyword_score` (с учетом anchor boost)
7. **Seed selection + expansion**: выбор `seed_k=3` лучших, расширение соседями (только с `keyword_score > 0`)
8. **Формирование сигнала yes/no**: определяет `yesno_signal` и `yesno_rationale` (только для `procedural / best practices`, не для `regulatory_principle`)

### Фильтры
- `granularity = 'atomic'`
- `part IN [164]` (по умолчанию, с fallback)

---

## 6. PermissionDisclosureRetriever

### Назначение
Для вопросов о разрешениях на раскрытие PHI. Возвращает набор атомарных параграфов из Part 164 (Privacy Rule), которые описывают разрешенные раскрытия, содержат условия, включают исключения/ограничения.

### Примеры вопросов
- "Can I disclose personal health information to family members?"
- "When can PHI be disclosed without authorization?"
- "Is disclosure to law enforcement permitted?"

### Входные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `question_embedding` | `List[float]` | - | Эмбеддинг вопроса (4096 dim) |
| `max_results` | `int` | `6` | Максимальное количество результатов |
| `question` | `Optional[str]` | `None` | Текст вопроса |
| `doc_id` | `Optional[str]` | `"hipaa-reg-2013-03-26"` | ID документа |
| `part` | `int` | `164` | Номер части (жестко Part 164) |
| `k` | `int` | `6` | Количество итоговых результатов |
| `seed_k` | `int` | `4` | Количество seed-чанков |
| `expand_refs` | `bool` | `True` | Расширять через references |
| `expand_siblings` | `bool` | `True` | Расширять через siblings |
| `max_ref_hops` | `int` | `1` | Максимальная глубина reference expansion |
| `ref_limit` | `int` | `12` | Количество ref-параграфов для подтягивания |
| `strict_quote` | `bool` | `True` | Требовать точную цитату |

### Выходные данные

```python
[
    {
        "chunk_id": "chk:164.502:(a)",
        "anchor": "§164.502(a)",
        "section_id": "sec:164.502",
        "section_number": "164.502",
        "section_title": "Uses and disclosures of protected health information: General rules",
        "paragraph_path": "(a)",
        "parent_chunk_id": "chk:164.502:section",
        "text_raw": "(a) Standard. A covered entity may not use or disclose protected health information...",
        "page_start": 200,
        "page_end": 201,
        "scores": {
            "vector_score": 0.5887,
            "fts_score": 0.5000,
            "keyword_score": 0.8000,
            "final_score": 0.5887
        },
        "flags": {
            "is_seed": True,
            "is_ref": False,
            "is_sibling": False
        },
        "explanation": "disclosure retrieval",
        "policy_signal": "prohibited"  # "permitted" | "prohibited" | "conditional" | "unclear"
    },
    ...
]
```

### Алгоритм
1. **Topic extraction**: извлекает тему раскрытия (family, law enforcement, public health, employer, business associate, research)
2. **FTS query**: усиление disclosure-токенами (disclose, disclosure, permit, authorization, minimum necessary) + topic-токенами
3. **Seed retrieval** (hybrid):
   - Фильтры: `granularity = 'atomic'`, `part = 164`
   - Fallback: если по Part 164 мало кандидатов, снимает фильтр part
4. **Clause / Evidence scoring**: усиление чанков с "правовой логикой"
   - +0.4 если есть may/is permitted, may disclose
   - +0.4 если есть except/subject to/only if
   - +0.4 если есть minimum necessary
   - +0.3 если есть may not/prohibited
   - +0.6 если есть topic tokens
5. **Merge + final scoring**: `final_score = 0.50*vector_score + 0.20*fts_score_norm + 0.30*keyword_score`
6. **Seed selection**: выбор `seed_k=4` лучших (разрешает 2 сида из одной секции для узких вопросов)
7. **Reference expansion**: расширение через `chunk_refs` (если таблица существует)
8. **Sibling expansion**: расширение соседями по `parent_chunk_id` или `section_id`
9. **Policy signal**: определяет `policy_signal` на основе найденных чанков

### Фильтры
- `granularity = 'atomic'`
- `part = 164` (жестко, с fallback)

---

## 7. CitationRetriever

### Назначение
Для вопросов, где явно требуется цитирование. Возвращает атомарные параграфы в виде anchor + text_raw (дословная цитата) без пересказа, интерпретации, выводов.

### Примеры вопросов
- "Cite the specific regulation texts regarding permitted disclosures to law enforcement."

### Входные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `question_embedding` | `List[float]` | - | Эмбеддинг вопроса (4096 dim) |
| `max_results` | `int` | `6` | Максимальное количество результатов |
| `question` | `Optional[str]` | `None` | Текст вопроса |
| `doc_id` | `Optional[str]` | `"hipaa-reg-2013-03-26"` | ID документа |
| `anchor_prefix` | `Optional[str]` | `None` | Префикс anchor (например, "§164.512") |
| `k` | `int` | `6` | Количество итоговых цитат |
| `seed_k` | `int` | `6` | Количество seed-чанков |
| `expand_section` | `bool` | `True` | Расширить внутри секции |
| `min_relevance` | `Optional[float]` | `None` | Минимальный порог final_score |

### Выходные данные

```python
[
    {
        "chunk_id": "chk:164.512:(f)",
        "anchor": "§164.512(f)",
        "text_raw": "(f) Standard: Disclosures for law enforcement purposes. A covered entity may disclose protected health information...",
        "section_id": "sec:164.512",
        "section_number": "164.512",
        "section_title": "Uses and disclosures for which an authorization or opportunity to agree or object is not required",
        "page_start": 250,
        "page_end": 251,
        "scores": {
            "vector_score": 0.7234,
            "fts_score": 0.5000,
            "final_score": 0.7234
        },
        "explanation": "citation retrieval"
    },
    ...
]
```

### Алгоритм
1. **Определение anchor prefix**: автоматически по вопросу (law enforcement → "§164.512") или использует дефолт
2. **Vector search** по atomic + hard filter по anchor prefix:
   - Фильтры: `granularity = 'atomic'`, `part = 164`, `anchor LIKE '§164.512%'`
3. **FTS search** внутри anchor scope (помогает выбрать нужные подпункты внутри §164.512)
4. **Merge + selection**: `final_score = 0.8*vector_score + 0.2*fts_score_norm`
5. **Coverage expansion**: подтягивает соседей по section_id в пределах anchor_like, если seeds покрывают мало подпунктов
6. **Сортировка** по anchor (чтобы подпункты шли в порядке документа)

### Фильтры
- `granularity = 'atomic'`
- `part = 164`
- `anchor LIKE '§164.512%'` (hard filter)

---

## 8. NavigationRetriever

### Назначение
Для навигационных вопросов типа "какая часть покрывает X". Возвращает структурные элементы (Part/Subpart/Section), а не цитаты текста.

### Примеры вопросов
- "Which part covers data privacy measures?"
- "Where are privacy requirements located?"
- "Which section is about applicability?"

### Входные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `question_embedding` | `List[float]` | - | Эмбеддинг вопроса (не используется) |
| `max_results` | `int` | `3` | Максимальное количество результатов |
| `question` | `Optional[str]` | `None` | Текст вопроса |
| `doc_id` | `Optional[str]` | `"hipaa-reg-2013-03-26"` | ID документа |
| `k_sections` | `int` | `10` | Количество кандидатов для поиска |
| `k_answer` | `int` | `3` | Количество итоговых секций для ответа |

### Выходные данные

```python
[
    {
        "part": 164,
        "subpart": None,
        "section_id": "sec:164.102",
        "section_number": "164.102",
        "section_title": "Statutory basis",
        "anchor": "§164.102",
        "page_start": 100,
        "page_end": 101,
        "scores": {
            "rule_score": 0.95,
            "title_score": 0.0,
            "final_score": 0.95
        },
        "explanation": "keyword privacy/disclosure/PHI -> Part 164 (Privacy Rule)"
    },
    ...
]
```

### Алгоритм
1. **Rule-based поиск**: проверка правил для быстрых маршрутов
   - privacy/disclosure/PHI → Part 164
   - security/encryption → Part 164
   - transactions/code sets → Part 162
   - general/applicability/definitions → Part 160
2. **Title-based поиск**: FTS по заголовкам секций + trigram fallback
3. **Сборка ответа**: Part + suggested entry points (1-3 секции внутри этого part)
4. **Scoring**: `final_score = max(rule_score, 0.6*title_score_norm)`

### Особенность
Возвращает структуру с `section_id` вместо `chunk_id`, поэтому в `app.py` используется специальная конвертация.

---

## Post-Classification Override

Система включает слой post-classification override, который применяет доменные правила после классификации вопросов LLM.

### Правила override

Если вопрос матчится под паттерн:
- "what does 'X' mean" / "what does X mean" / "define X" / "what is X" / "explain X"

И `X` в словаре регуляторных концепций:
- minimum necessary
- reasonable safeguards
- addressable implementation specification
- administrative safeguards
- technical safeguards
- physical safeguards
- reasonable and appropriate
- covered entity
- business associate
- protected health information
- phi
- individually identifiable health information

То категория переопределяется на `regulatory_principle` (если изначально была `definition`).

### Модуль
- Файл: `classification_override.py`
- Функция: `apply_classification_override(classification, question) -> QuestionClassification`

### Использование
Override применяется автоматически в `app.py` после классификации и перед выбором ретривера.

---

## 9. GeneralRetriever

### Назначение
Fallback ретривер для прочих вопросов. Обрабатывает любые "прочие" вопросы, которые не подходят под специализированные ретриверы. Возвращает достаточный контекст с цитируемостью (anchors), используя гибридный поиск и иерархическую структуру (atomic + родительская секция).

### Примеры вопросов
- "What are the general requirements for HIPAA compliance?"
- "Tell me about HIPAA in general"
- Любые другие вопросы, не подходящие под специализированные категории

### Входные параметры

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `question_embedding` | `List[float]` | - | Эмбеддинг вопроса (4096 dim) |
| `max_results` | `int` | `8` | Максимальное количество результатов |
| `question` | `Optional[str]` | `None` | Текст вопроса |
| `doc_id` | `Optional[str]` | `"hipaa-reg-2013-03-26"` | ID документа |
| `k` | `int` | `8` | Количество итоговых результатов |
| `seed_k` | `int` | `6` | Количество первичных кандидатов |
| `fts_weight` | `float` | `0.35` | Вес FTS score |
| `vector_weight` | `float` | `0.65` | Вес vector score |
| `include_parent_sections` | `bool` | `True` | Добавлять parent/section chunks |
| `parent_limit` | `int` | `2` | Максимум уникальных parent chunks |
| `max_per_section` | `int` | `2` | Максимум atomic чанков из одной section_id |
| `use_part_hint` | `bool` | `True` | Использовать part hint |

### Выходные данные

```python
[
    {
        "chunk_id": "chk:160.310:(a)",
        "anchor": "§160.310(a)",
        "section_id": "sec:160.310",
        "section_number": "160.310",
        "section_title": "Responsibilities of covered entities and business associates",
        "granularity": "atomic",
        "paragraph_path": "(a)",
        "parent_chunk_id": "chk:160.310:section",
        "text_raw": "(a) General rule...",
        "page_start": 50,
        "page_end": 51,
        "scores": {
            "vector_score": 0.5000,
            "fts_score": 0.3000,
            "final_score": 0.4450
        },
        "flags": {
            "is_seed": True,
            "is_parent": False,
            "is_context": False
        },
        "explanation": "general retrieval",
        "coverage_note": "Top contexts span 5 sections; use anchors for citations."
    },
    {
        "chunk_id": "chk:162.600:section",
        "anchor": "§162.600",
        "section_id": "sec:162.600",
        "section_number": "162.600",
        "section_title": "General requirements",
        "granularity": "section",
        "text_raw": "§162.600 General requirements...",
        "scores": {
            "vector_score": 0.4500,
            "fts_score": 0.0,
            "final_score": 0.2925
        },
        "flags": {
            "is_seed": False,
            "is_parent": True,
            "is_context": False
        },
        "explanation": "parent section context"
    },
    ...
]
```

### Алгоритм
1. **Part hint**: определяет part hint из вопроса (privacy/disclosure/PHI → 164, transactions → 162)
2. **Candidate retrieval** (hybrid) по atomic:
   - Fallback: если с part_hint мало кандидатов, пробует без него
   - Fallback 2: если atomic не дал результатов, пробует section chunks
3. **Merge + scoring**: `final_score = vector_weight*vector_score + fts_weight*fts_score_norm`
4. **Diversity constraint**: выбирает diverse seeds с ограничением `max_per_section=2` из одной `section_id`
5. **Context enrichment**: добавляет parent/section chunks для enrichment контекста
6. **Final assembly**: дедупликация и сортировка (section перед atomic, затем по anchor)

### Фильтры
- `granularity = 'atomic'` (основной поиск)
- `part` (опционально, soft boost)

---

## 10. SemanticRetriever

### Назначение
Базовый семантический ретривер (заглушка). Планируется для использования как fallback, но пока не реализован.

### Статус
⚠️ **Заглушка** - возвращает пустой список. Будет реализован позже.

---

## API Эндпойнты

### POST /search

Поиск релевантных чанков по вопросу. Включает классификацию вопроса и автоматический выбор ретривера.

**Запрос:**
```json
{
    "question": "What does business associate mean?",
    "max_results": 5
}
```

**Ответ:**
```json
{
    "question": "What does business associate mean?",
    "classification": {
        "category": "definition",
        "confidence": 0.95,
        "reasoning": "Question asks for definition of a term"
    },
    "retrieved_chunks": [
        {
            "chunk_id": "chk:160.103:Business_associate(1)",
            "section_number": "160.103",
            "section_title": "Definitions",
            "text_raw": "Business associate: (1) Except as provided...",
            "similarity": 0.8390,
            "anchor": "§160.103:Business_associate(1)"
        }
    ],
    "total_found": 1
}
```

### POST /answer

Генерация полного ответа на вопрос. Включает классификацию, поиск релевантных чанков и генерацию ответа через LLM.

**Запрос:**
```json
{
    "question": "What does business associate mean?",
    "max_results": 5
}
```

**Ответ:**
```json
{
    "question": "What does business associate mean?",
    "classification": {
        "category": "definition",
        "confidence": 0.95,
        "reasoning": "Question asks for definition of a term"
    },
    "retrieved_chunks": [...],
    "answer": "According to §160.103, a business associate means...",
    "sources": ["chk:160.103:Business_associate(1)"]
}
```

---

## Контракты ретриверов

### Общий контракт (BaseRetriever)

Все ретриверы должны возвращать список словарей со следующими обязательными полями:

```python
{
    "chunk_id": str,              # Уникальный ID чанка
    "anchor": Optional[str],       # Anchor для цитирования (например, "§160.103")
    "section_id": str,            # ID секции
    "section_number": str,         # Номер секции (например, "160.103")
    "section_title": str,          # Заголовок секции
    "text_raw": str,               # Текст чанка (дословная цитата)
    "scores": {
        "vector_score": float,     # Оценка векторного поиска (0..1)
        "fts_score": float,         # Оценка FTS поиска (0..1)
        "final_score": float       # Финальная оценка (0..1)
    }
}
```

### Специфические поля по ретриверам

| Ретривер | Дополнительные поля |
|----------|---------------------|
| `DefinitionRetriever` | `term`, `explanation` |
| `ScopeRetriever` | `paragraph_path`, `explanation` |
| `PenaltiesRetriever` | `paragraph_path`, `scores.amount_score`, `explanation` |
| `ProceduralRetriever` | `paragraph_path`, `scores.keyword_score`, `yesno_signal`, `yesno_rationale`, `explanation` |
| `PermissionDisclosureRetriever` | `paragraph_path`, `parent_chunk_id`, `scores.keyword_score`, `flags`, `policy_signal`, `explanation` |
| `CitationRetriever` | `explanation` (минимальный набор, только anchor + text_raw) |
| `NavigationRetriever` | `part`, `subpart`, `scores.rule_score`, `scores.title_score`, `explanation` (возвращает `section_id` вместо `chunk_id`) |
| `GeneralRetriever` | `granularity`, `paragraph_path`, `parent_chunk_id`, `flags`, `coverage_note`, `explanation` |

---

## Использование в коде

### Прямой вызов ретривера

```python
from retrievers import DefinitionRetriever
from embeddings import get_embedding_client

# Создание ретривера
retriever = DefinitionRetriever()

# Создание эмбеддинга
embedding_client = get_embedding_client()
question_embedding = embedding_client.create_embedding("What does business associate mean?")

# Вызов ретривера
results = await retriever.retrieve(
    question_embedding=question_embedding,
    question="What does business associate mean?",
    max_results=1,
    k=1
)
```

### Автоматический выбор через get_retriever_for_category

```python
from retrievers import get_retriever_for_category
from embeddings import get_embedding_client

# Автоматический выбор ретривера
retriever = get_retriever_for_category(
    category="definition",
    question="What does business associate mean?"
)

# Использование
embedding_client = get_embedding_client()
question_embedding = embedding_client.create_embedding("What does business associate mean?")

results = await retriever.retrieve(
    question_embedding=question_embedding,
    question="What does business associate mean?",
    max_results=1
)
```

### Использование через FastAPI

Ретриверы автоматически вызываются через эндпойнты `/search` и `/answer`:

```bash
# Поиск чанков
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does business associate mean?",
    "max_results": 5
  }'

# Генерация ответа
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does business associate mean?",
    "max_results": 5
  }'
```

---

## Алгоритмы поиска

### Гибридный поиск (Hybrid Search)

Большинство ретриверов используют комбинацию:
- **Vector Search**: семантический поиск по косинусному сходству эмбеддингов
- **FTS (Full-Text Search)**: лексический поиск по тексту с использованием PostgreSQL `tsvector`/`tsquery`

**Формула объединения:**
```
final_score = α * vector_score + β * fts_score_norm
```

Где `α` и `β` зависят от ретривера:
- `OverviewPurposeRetriever`: 0.7 / 0.3
- `DefinitionRetriever`: 0.6 / 0.4 (если term есть) или 0.8 / 0.2
- `ScopeRetriever`: 0.7 / 0.3
- `PenaltiesRetriever`: 0.65 / 0.35 (base_score), затем + amount_score
- `ProceduralRetriever`: 0.55 / 0.20 / 0.25 (vector / fts / keyword)
- `PermissionDisclosureRetriever`: 0.50 / 0.20 / 0.30 (vector / fts / keyword)
- `CitationRetriever`: 0.8 / 0.2
- `GeneralRetriever`: 0.65 / 0.35 (настраиваемо)

### Расширение результатов (Expansion)

Многие ретриверы используют расширение результатов:
- **Sibling expansion**: подтягивание соседних чанков по `parent_chunk_id` или `section_id`
- **Reference expansion**: подтягивание связанных чанков через таблицу `chunk_refs` (если существует)
- **Coverage expansion**: подтягивание соседей для сбора полного списка подпунктов

### Дедупликация и сортировка

Все ретриверы выполняют:
- **Дедупликацию** по `chunk_id`
- **Сортировку** по `section_id`, затем по `anchor` (для упорядочивания как в документе)

---

## Рекомендации по использованию

1. **Для определений**: используйте `DefinitionRetriever` - возвращает точные определения с anchor'ами
2. **Для навигации**: используйте `NavigationRetriever` - возвращает структурные элементы (Part/Section)
3. **Для штрафов**: используйте `PenaltiesRetriever` - усиливает чанки с суммами/диапазонами
4. **Для процедур**: используйте `ProceduralRetriever` - формирует yes/no сигнал для answer-слоя
5. **Для раскрытий**: используйте `PermissionDisclosureRetriever` - возвращает условия и исключения
6. **Для цитирования**: используйте `CitationRetriever` - строго anchor + text_raw без пересказа
7. **Для прочих вопросов**: используйте `GeneralRetriever` - универсальный fallback с coverage

---

## Примечания

- Все ретриверы работают с базой данных PostgreSQL + pgvector
- Эмбеддинги генерируются через VseGPT API (модель `emb-qwen/qwen3-embedding-8b`, размерность 4096)
- FTS использует PostgreSQL `to_tsvector`/`plainto_tsquery` с языком 'english'
- Vector search использует оператор `<=>` (косинусное расстояние) из pgvector
- Все ретриверы логируют процесс работы через `logger` для отладки
