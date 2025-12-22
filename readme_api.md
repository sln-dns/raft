# HIPAA Regulations RAG API Documentation

## Обзор

FastAPI приложение для поиска и генерации ответов по HIPAA регуляциям. API предоставляет три основных эндпойнта для работы с вопросами пользователей.

**Base URL:** `http://localhost:8000` (или `http://127.0.0.1:8000`)

**Version:** 0.1.0

---

## Эндпойнты

### 1. `GET /` - Корневой эндпойнт

**Описание:** Возвращает информацию об API и доступных эндпойнтах.

**Метод:** `GET`

**Параметры:** Нет

**Пример запроса:**
```bash
curl http://127.0.0.1:8000/
```

**Пример ответа:**
```json
{
  "message": "HIPAA Regulations RAG API",
  "version": "0.1.0",
  "endpoints": {
    "/classify": "Классификация вопроса",
    "/search": "Поиск релевантных чанков",
    "/answer": "Полный ответ с генерацией"
  }
}
```

**Статус коды:**
- `200 OK` - Успешный запрос

---

### 2. `GET /health` - Проверка здоровья

**Описание:** Проверяет состояние API и доступность основных компонентов (классификатор, embedding client).

**Метод:** `GET`

**Параметры:** Нет

**Пример запроса:**
```bash
curl http://127.0.0.1:8000/health
```

**Пример ответа:**
```json
{
  "status": "healthy",
  "classifier": true,
  "embedding_client": true
}
```

**Поля ответа:**
- `status` (string): Статус API. Всегда `"healthy"` если API работает.
- `classifier` (boolean): `true` если классификатор инициализирован.
- `embedding_client` (boolean): `true` если embedding client инициализирован.

**Статус коды:**
- `200 OK` - API работает нормально

**Использование:** Используйте этот эндпойнт для проверки доступности API перед отправкой запросов.

---

### 3. `POST /classify` - Классификация вопроса

**Описание:** Классифицирует вопрос пользователя, определяя его категорию для выбора подходящего ретривера. Применяет post-classification override для доменных правил.

**Метод:** `POST`

**Content-Type:** `application/json`

**Тело запроса:**
```json
{
  "question": "string",      // Обязательно, минимум 1 символ
  "max_results": 5           // Опционально, по умолчанию 5, диапазон 1-20
}
```

**Валидация:**
- `question`: обязательное поле, минимум 1 символ
- `max_results`: опционально, по умолчанию 5, должно быть от 1 до 20

**Пример запроса:**
```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"question": "What does minimum necessary mean?", "max_results": 5}'
```

**Пример ответа:**
```json
{
  "category": "regulatory_principle",
  "confidence": 1.0,
  "reasoning": "Override: 'minimum necessary' is a regulatory principle/concept, not a simple definition. Original: Вопрос запрашивает определение термина 'minimum necessary', что соответствует категории определения терминов."
}
```

**Поля ответа:**
- `category` (string): Категория вопроса. Возможные значения:
  - `"definition"` - Определения терминов
  - `"overview / purpose"` - Общий обзор и цель
  - `"overview / navigation"` - Навигация по структуре документа
  - `"scope / applicability"` - Применимость и область действия
  - `"penalties"` - Штрафы и наказания
  - `"procedural / best practices"` - Процедуры и лучшие практики
  - `"permission / disclosure"` - Разрешения на раскрытие информации
  - `"citation-required"` - Требуется цитирование
  - `"regulatory_principle"` - Регуляторные принципы и концепции (после override)
  - `"other"` - Прочие вопросы
- `confidence` (float): Уверенность классификации от 0.0 до 1.0
- `reasoning` (string): Обоснование классификации (может содержать информацию об override)

**Ошибки:**
- `422 Unprocessable Entity` - Ошибка валидации:
  ```json
  {
    "detail": [
      {
        "type": "string_too_short",
        "loc": ["body", "question"],
        "msg": "String should have at least 1 character",
        "input": "",
        "ctx": {"min_length": 1}
      }
    ]
  }
  ```
- `500 Internal Server Error` - Ошибка классификации:
  ```json
  {
    "detail": "Ошибка классификации: <описание ошибки>"
  }
  ```

**Особенности:**
- Применяется post-classification override: если вопрос соответствует паттерну "what does X mean" и X входит в список регуляторных концепций (например, "minimum necessary", "reasonable safeguards"), категория может быть изменена на `regulatory_principle`.
- **Важно:** `regulatory_principle` означает, что термин может не иметь формального определения в разделе Definitions; ответ цитирует регулирующие положения, где термин используется как нормативный принцип/требование.

---

### 4. `POST /search` - Поиск релевантных чанков

**Описание:** Выполняет поиск релевантных чанков по вопросу. Включает классификацию вопроса, создание эмбеддинга и семантический поиск в базе данных с использованием специализированного ретривера.

**Метод:** `POST`

**Content-Type:** `application/json`

**Тело запроса:**
```json
{
  "question": "string",      // Обязательно, минимум 1 символ
  "max_results": 5           // Опционально, по умолчанию 5, диапазон 1-20
}
```

**Пример запроса:**
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "What does business associate mean?", "max_results": 2}'
```

**Примечание:** В этом примере `business associate` является формально определенным термином (category: `definition`), поэтому поиск возвращает чанки из раздела Definitions (§160.103).

**Пример ответа:**
```json
{
  "question": "What does business associate mean?",
  "classification": {
    "category": "definition",
    "confidence": 0.9,
    "reasoning": "Вопрос запрашивает определение термина 'business associate', что соответствует категории определения терминов."
  },
  "retrieved_chunks": [
    {
      "chunk_id": "chk:160.103:def:Business_associate_1",
      "section_number": "160.103",
      "section_title": "Definitions",
      "text_raw": "Business associate: (1) Except as provided in paragraph (4) of this definition, business associate means, with respect to a covered entity, a person who:\n\n(i) On behalf of such covered entity or of an organized health care arrangement (as defined in this section) in which the covered entity participates, but other than in the capacity of a member of the workforce of such covered entity or arrangement, creates, receives, maintains, or transmits protected health information for a function or activity regulated by this subchapter, including claims processing or administration, data analysis, processing or administration, utilization review, quality assurance, patient safety activities listed at 42 CFR 3.20, billing, benefit management, practice management, and repricing; or\n\n(ii) Provides, other than in the capacity of a member of the workforce of such covered entity, legal, actuarial, accounting, consulting, data aggregation (as defined in § 164.501 of this subchapter), management, administrative, accreditation, or financial services to or for such covered entity, or to or for an organized health care arrangement in which the covered entity participates, where the provision of the service involves the disclosure of protected health information from such covered entity or arrangement, or from another business associate of such covered entity or arrangement, to the person.",
      "similarity": 0.8389972824053118,
      "anchor": "§160.103:Business_associate(1)",
      "chunk_kind": null,
      "granularity": null
    }
  ],
  "total_found": 1
}
```

**Поля ответа:**
- `question` (string): Оригинальный вопрос пользователя
- `classification` (object): Результат классификации (см. `/classify`)
- `retrieved_chunks` (array): Массив найденных чанков. Каждый чанк содержит:
  - `chunk_id` (string): Уникальный идентификатор чанка
  - `section_number` (string): Номер секции (например, "160.103")
  - `section_title` (string): Заголовок секции
  - `text_raw` (string): Текст чанка (дословная цитата из документа)
  - `similarity` (float): Оценка релевантности от 0.0 до 1.0
  - `anchor` (string, optional): Якорь для цитирования (например, "§160.103:Business_associate(1)")
  - `chunk_kind` (string, optional): Тип чанка
  - `granularity` (string, optional): Гранулярность чанка
- `total_found` (integer): Общее количество найденных чанков

**Особенности:**
- Для категории `overview / navigation` возвращаются результаты с `section_id` вместо `chunk_id`, и `text_raw` формируется из метаданных секции.
- Количество возвращаемых чанков может быть меньше `max_results`, если найдено меньше релевантных результатов.

**Ошибки:**
- `422 Unprocessable Entity` - Ошибка валидации (см. `/classify`)
- `500 Internal Server Error` - Ошибка поиска:
  ```json
  {
    "detail": "Ошибка поиска: <описание ошибки>"
  }
  ```

---

### 5. `POST /answer` - Полный ответ с генерацией

**Описание:** Генерирует полный ответ на вопрос. Включает классификацию, поиск релевантных чанков и генерацию ответа с использованием LLM. Это основной эндпойнт для получения готовых ответов.

**Метод:** `POST`

**Content-Type:** `application/json`

**Тело запроса:**
```json
{
  "question": "string",      // Обязательно, минимум 1 символ
  "max_results": 5           // Опционально, по умолчанию 5, диапазон 1-20
}
```

**Примеры запросов и ответов:**

**Пример 1: Definition (формально определенный термин)**
```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What does business associate mean?", "max_results": 2}'
```

```json
{
  "question": "What does business associate mean?",
  "classification": {
    "category": "definition",
    "confidence": 0.9,
    "reasoning": "Вопрос запрашивает определение термина 'business associate', что соответствует категории определения терминов."
  },
  "retrieved_chunks": [
    {
      "chunk_id": "chk:160.103:def:Business_associate_1",
      "section_number": "160.103",
      "section_title": "Definitions",
      "text_raw": "Business associate: (1) Except as provided in paragraph (4) of this definition, business associate means, with respect to a covered entity, a person who:\n\n(i) On behalf of such covered entity...",
      "similarity": 0.8389972824053118,
      "anchor": "§160.103:Business_associate(1)",
      "chunk_kind": null,
      "granularity": null
    }
  ],
  "answer": "According to §160.103, a business associate means, with respect to a covered entity, a person who: (i) On behalf of such covered entity or of an organized health care arrangement...",
  "sources": [
    "§160.103:Business_associate(1)"
  ],
  "debug": {
    "model": "anthropic/claude-haiku-4.5",
    "chunks_count": 1,
    "category": "definition",
    "confidence": 0.9,
    "prompt_template": "build_prompt",
    "answer_policy": "quoted_answer",
    "citations_validated": true,
    "valid_citations_count": 1
  }
}
```

**Пример 2: Regulatory Principle (нормативный принцип, не формальное определение)**
```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What does minimum necessary mean?", "max_results": 3}'
```

```json
{
  "question": "What does minimum necessary mean?",
  "classification": {
    "category": "regulatory_principle",
    "confidence": 1.0,
    "reasoning": "Override: 'minimum necessary' is a regulatory principle/concept, not a simple definition. Original: Вопрос запрашивает определение термина 'minimum necessary', что соответствует категории определения терминов."
  },
  "retrieved_chunks": [
    {
      "chunk_id": "chk:164.502:(b)",
      "section_number": "164.502",
      "section_title": "Uses and disclosures of protected health information: General rules",
      "text_raw": "(b) Standard: Minimum necessary\n\n(1) Minimum necessary applies. When using or disclosing protected health information or when requesting protected health information from another covered entity or business associate, a covered entity or business associate must make reasonable efforts to limit protected health information to the minimum necessary to accomplish the intended purpose of the use, disclosure, or request.",
      "similarity": 0.7506185550866642,
      "anchor": "§164.502(b)",
      "chunk_kind": null,
      "granularity": null
    }
  ],
  "answer": "HIPAA does not provide a standalone definition of 'minimum necessary' in the Definitions section. However, §164.502(b) establishes it as a standard requiring covered entities and business associates to make reasonable efforts to limit protected health information to the minimum necessary to accomplish the intended purpose of the use, disclosure, or request.",
  "sources": [
    "§164.502(b)",
    "§164.514(d)"
  ],
  "debug": {
    "model": "anthropic/claude-haiku-4.5",
    "chunks_count": 3,
    "category": "regulatory_principle",
    "confidence": 1.0,
    "prompt_template": "build_prompt",
    "answer_policy": "quoted_answer",
    "citations_validated": true,
    "valid_citations_count": 2
  }
}
```

**Пример 3: Citation-Required (строгое цитирование, LLM пропущен)**
```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Cite the specific regulation texts regarding permitted disclosures to law enforcement.", "max_results": 5}'
```

```json
{
  "question": "Cite the specific regulation texts regarding permitted disclosures to law enforcement.",
  "classification": {
    "category": "citation-required",
    "confidence": 0.9,
    "reasoning": "Вопрос требует ссылки на конкретные тексты регуляций, касающиеся разрешенных раскрытий информации правоохранительным органам."
  },
  "retrieved_chunks": [
    {
      "chunk_id": "chk:164.512:(f)",
      "section_number": "164.512",
      "section_title": "Uses and disclosures for which an authorization or opportunity to agree or object is not required",
      "text_raw": "(f) Standard: Disclosures for law enforcement purposes. A covered entity may disclose protected health information for a law enforcement purpose to a law enforcement official if the conditions in paragraphs (f)(1) through (f)(6) of this section are met, as applicable.",
      "similarity": 0.7234394312785076,
      "anchor": "§164.512(f)",
      "chunk_kind": null,
      "granularity": null
    }
  ],
  "answer": "§164.512(f) - (f) Standard: Disclosures for law enforcement purposes. A covered entity may disclose protected health information for a law enforcement purpose to a law enforcement official if the conditions in paragraphs (f)(1) through (f)(6) of this section are met, as applicable.\n\n§164.512(a) - (a) Standard: Uses and disclosures required by law...",
  "sources": [
    "§164.512(f)",
    "§164.512(a)",
    "§164.512(e)"
  ],
  "debug": {
    "policy": "strict_citation",
    "llm_skipped": true,
    "citations_count": 5,
    "context_items_count": 5
  }
}
```

**Поля ответа:**
- `question` (string): Оригинальный вопрос пользователя
- `classification` (object): Результат классификации (см. `/classify`)
- `retrieved_chunks` (array): Массив найденных чанков (см. `/search`)
- `answer` (string): Сгенерированный ответ на вопрос. Может содержать:
  - Текст ответа с цитатами
  - "Insufficient context to provide exact citation." - если контекста недостаточно
  - Для `citation-required`: прямые цитаты из документа с якорями
  - Для `procedural / best practices`: ответы формата "YES/NO/UNCLEAR" с обоснованием
- `sources` (array): Список источников для цитирования. **Приоритет:** якоря (anchors) всегда предпочтительнее; `chunk_id` используется только как fallback, если якорь отсутствует.
  - **Предпочтительно:** Якоря (например, "§164.512(a)", "§160.103:Business_associate(1)")
  - **Fallback:** `chunk_id` (внутренний идентификатор, используется только если anchor отсутствует)
- `debug` (object, optional): Отладочная информация:
  - `model` (string): Использованная LLM модель
  - `chunks_count` (integer): Количество чанков в контексте
  - `category` (string): Категория вопроса
  - `confidence` (float): Уверенность классификации
  - `prompt_template` (string): Использованный шаблон промпта
  - `answer_policy` (string): Политика генерации ответа:
    - `"strict_citation"` - Строгое цитирование (LLM пропущен)
    - `"navigation"` - Навигация по структуре
    - `"summary"` - Краткое резюме
    - `"quoted_answer"` - Ответ с цитатами
    - `"listing"` - Список пунктов
  - `permission_policy` (string, optional): Политика разрешений (для `permission / disclosure`):
    - `"permitted"` - Разрешено
    - `"prohibited"` - Запрещено
    - `"conditional"` - Условно
    - `"unclear"` - Неясно
  - `citations_validated` (boolean): Были ли валидированы цитаты
  - `valid_citations_count` (integer): Количество валидных цитат
  - `retriever_yesno_signal` (string, optional): Сигнал от ретривера для `procedural / best practices`:
    - `"yes"` - Да
    - `"no"` - Нет
    - `"unclear"` - Неясно
  - `retriever_yesno_rationale` (string, optional): Обоснование сигнала
  - `retriever_policy_signal` (string, optional): Сигнал от ретривера для `permission / disclosure`
  - `llm_skipped` (boolean, optional): Был ли пропущен LLM (для `strict_citation`)
  - `citations_count` (integer, optional): Количество цитат (для `strict_citation`)

**Особенности по категориям:**

1. **`citation-required`**: 
   - LLM пропускается (`llm_skipped: true`)
   - Ответ содержит прямые цитаты из документа с якорями
   - Формат: `§164.512(a) - (текст цитаты)`

2. **`procedural / best practices`**:
   - Ответ может начинаться с "YES", "NO", "UNCLEAR" или "PARTIALLY"
   - Содержит обоснование и цитаты
   - Может включать `retriever_yesno_signal` в debug

3. **`permission / disclosure`**:
   - Ответ описывает условия разрешения/запрета
   - Может включать `retriever_policy_signal` в debug

4. **`regulatory_principle`**:
   - **Важно:** Термин может не иметь формального определения в разделе Definitions; ответ цитирует регулирующие положения, где термин используется как нормативный принцип/требование.
   - Ответ начинается с "HIPAA does not provide a standalone definition of 'X'..."
   - Объясняет нормативное значение как принцип/требование, а не словарное определение
   - Содержит 1-3 цитаты с якорями из соответствующих разделов (например, §164.502(b) для "minimum necessary")

5. **`overview / navigation`**:
   - Ответ указывает Part/Subpart/Section
   - Может содержать suggested entry points

**Ошибки:**
- `422 Unprocessable Entity` - Ошибка валидации:
  ```json
  {
    "detail": [
      {
        "type": "string_too_short",
        "loc": ["body", "question"],
        "msg": "String should have at least 1 character",
        "input": "",
        "ctx": {"min_length": 1}
      }
    ]
  }
  ```
  Или:
  ```json
  {
    "detail": [
      {
        "type": "greater_than_equal",
        "loc": ["body", "max_results"],
        "msg": "Input should be greater than or equal to 1",
        "input": 0,
        "ctx": {"ge": 1}
      }
    ]
  }
  ```
- `500 Internal Server Error` - Ошибка генерации ответа:
  ```json
  {
    "detail": "Ошибка генерации ответа: <описание ошибки>"
  }
  ```

**Время выполнения:**
- Обычно 5-60 секунд в зависимости от сложности вопроса и количества чанков
- Для `citation-required` обычно быстрее (1-5 секунд), так как LLM пропускается

---

## Общие замечания

### CORS
API настроен с CORS middleware, разрешающим запросы с любых источников (`allow_origins=["*"]`). В продакшене следует указать конкретные домены.

### Формат ошибок
Все ошибки валидации возвращаются в формате Pydantic с детальной информацией о поле и типе ошибки.

### Логирование
API логирует все запросы и ошибки. Уровень логирования: `INFO`.

### Инициализация
При запуске приложения инициализируются:
- Классификатор вопросов
- Embedding client для создания векторных представлений

Если инициализация не удалась, `/health` вернет `false` для соответствующих компонентов.

---

## Примеры использования

### Пример 1: Простая классификация (с override)
```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"question": "What does minimum necessary mean?"}'
```
**Результат:** `category: "regulatory_principle"` (override применен, так как "minimum necessary" - регуляторный принцип, а не формальное определение)

### Пример 2: Поиск чанков
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "Which entities are regulated under HIPAA?", "max_results": 5}'
```

### Пример 3: Полный ответ
```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the penalties for HIPAA violations?", "max_results": 5}'
```

### Пример 4: Строгое цитирование (LLM пропущен)
```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "Cite the specific regulation texts regarding permitted disclosures to law enforcement.", "max_results": 5}'
```
**Результат:** Прямые цитаты из документа с якорями, LLM не используется (`llm_skipped: true` в debug)

---

## Технические детали

### Модели данных

**QuestionRequest:**
```python
{
  "question": str,        # min_length=1
  "max_results": int      # Optional, default=5, ge=1, le=20
}
```

**RetrievedChunk:**
```python
{
  "chunk_id": str,
  "section_number": str,
  "section_title": str,
  "text_raw": str,
  "similarity": float,    # ge=0.0, le=1.0
  "anchor": str | None,
  "chunk_kind": str | None,
  "granularity": str | None
}
```

**ClassificationResponse:**
```python
{
  "category": str,
  "confidence": float,
  "reasoning": str
}
```

---

## Версионирование

Текущая версия API: **0.1.0**

Изменения в API будут документироваться в этом файле с указанием версии.
