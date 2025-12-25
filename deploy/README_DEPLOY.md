# Деплой RAG системы для HIPAA регуляций

Этот документ описывает процесс деплоя приложения с использованием Docker Compose.

## Структура на сервере

На сервере в одной папке (например `/opt/raft/`) должны быть следующие файлы:

```
/opt/raft/
├── .env                  # Переменные окружения (секреты)
├── docker-compose.yml    # Файл compose (скопировать из deploy/)
├── raft.dump            # Дамп базы данных (pg_dump -Fc format)
└── README_DEPLOY.md     # Эта инструкция (опционально)
```

## Требования

- Docker
- Docker Compose
- Дамп базы данных в формате custom format (`pg_dump -Fc`)

## Подготовка

### 1. Создание дампа базы данных

На локальной машине (где работает база):

```bash
pg_dump -Fc -h localhost -U raft_user -d raft -f raft.dump
```

### 2. Копирование файлов на сервер

Скопируйте на сервер:
- `raft.dump` - дамп базы данных
- `deploy/docker-compose.yml` → `docker-compose.yml`
- `deploy/.env.example` → `.env` (и заполните секреты)

### 3. Настройка .env

Отредактируйте `.env` файл:

```bash
# GitHub Repository
GIT_REPO_URL=https://github.com/sln-dns/raft.git
GIT_BRANCH=main
# GIT_TOKEN=ghp_xxxxxxxxxxxxx  # Для private репозиториев

# PostgreSQL
POSTGRES_DB=raft
POSTGRES_USER=raft_user
POSTGRES_PASSWORD=your_secure_password_here

# API Keys
VSEGPT_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
CLASSIFICATION_MODEL=anthropic/claude-3-haiku
GENERATION_MODEL=anthropic/claude-3-haiku
EMBEDDING_MODEL=emb-qwen/qwen3-embedding-8b

# Application
DOC_ID=hipaa-reg-2013-03-26
```

## Запуск

### Первый запуск

```bash
docker compose up -d
```

Это выполнит:
1. Клонирование/обновление репозитория из GitHub
2. Запуск PostgreSQL
3. Восстановление базы данных из дампа (один раз)
4. Запуск API сервиса
5. Запуск UI сервиса
6. Запуск Nginx reverse proxy
7. Запуск Cloudflare Quick Tunnel

### Проверка статуса

```bash
# Статус всех сервисов
docker compose ps

# Логи всех сервисов
docker compose logs -f

# Логи конкретного сервиса
docker compose logs -f api
docker compose logs -f db-init
docker compose logs -f cloudflared
```

### Получение публичного URL

Cloudflare Quick Tunnel создает публичный HTTPS URL. Чтобы получить его:

```bash
docker logs -f raft-cloudflared
```

Ищите строку вида:
```
https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.trycloudflare.com
```

## Обновление кода

После push в GitHub, чтобы обновить код на сервере:

```bash
docker compose restart repo-sync api ui
```

Это обновит код из репозитория и перезапустит сервисы. База данных не затрагивается.

Подробнее:
1. `repo-sync` - обновит код в volume `/repo`
2. `api` и `ui` - перезапустятся с обновленным кодом

## Структура сервисов

### repo-sync
- Клонирует/обновляет код из GitHub в volume `repo:/repo`
- Запускается один раз при `docker compose up`
- Для обновления кода: `docker compose restart repo-sync`

### db
- PostgreSQL 16 с расширением pgvector
- Данные хранятся в volume `pgdata:/var/lib/postgresql/data`
- Порт 5432 открыт для отладки (можно закрыть в production)

### db-init
- Восстанавливает базу данных из `raft.dump`
- Выполняется один раз (проверяет marker файл)
- Marker файл: `/state/RESTORED_OK` в volume `dbinit_state`

### api
- FastAPI приложение
- Код монтируется из volume `repo:/repo:ro` (read-only)
- Использует зависимости из Docker образа
- Health check: `GET /health`

### ui
- Streamlit приложение (`ui_simple.py`)
- Код монтируется из volume `repo:/repo:ro` (read-only)
- Подключается к API по внутреннему адресу `http://api:8000`

### nginx
- Reverse proxy
- `/api/` → `api:8000`
- `/` → `ui:8501`
- Порт 80 открыт внутри Docker сети

### cloudflared
- Cloudflare Quick Tunnel
- Публикует `nginx:80` наружу через HTTPS
- Генерирует случайный URL вида `https://xxx.trycloudflare.com`

## Volumes

- `repo` - код репозитория (обновляется через repo-sync)
- `pgdata` - данные PostgreSQL (постоянные)
- `dbinit_state` - состояние инициализации БД (marker файл)

## Bind Mounts

- `./raft.dump:/backup/raft.dump:ro` - дамп базы данных (read-only)

## Troubleshooting

### Проблема: db-init не восстанавливает базу

Проверьте логи:
```bash
docker compose logs db-init
```

Убедитесь, что:
- Файл `raft.dump` существует в текущей директории
- PostgreSQL готов (healthcheck пройден)
- Marker файл не существует (если нужно пересоздать)

Для пересоздания базы:
```bash
docker compose down
docker volume rm raft_dbinit_state
docker compose up -d db db-init
```

### Проблема: API не подключается к базе

Проверьте:
1. Статус базы: `docker compose ps db`
2. Логи API: `docker compose logs api`
3. Переменную `DATABASE_URL` в `.env`

### Проблема: UI не видит API

Проверьте:
1. `API_BASE_URL` должен быть `http://api:8000` (внутренний адрес)
2. Статус API: `docker compose ps api`
3. Логи nginx: `docker compose logs nginx`

### Проблема: repo-sync не клонирует репозиторий

Проверьте:
1. `GIT_REPO_URL` в `.env`
2. Для private репозиториев: установите `GIT_TOKEN`
3. Логи: `docker compose logs repo-sync`

### Проблема: Cloudflare Tunnel не работает

Проверьте логи:
```bash
docker logs raft-cloudflared
```

Если URL не появляется, перезапустите:
```bash
docker compose restart cloudflared
```

## Безопасность

⚠️ **Важно:**

1. Файл `.env` содержит секреты - **НЕ коммитьте его в git**
2. Дамп базы содержит данные - храните его безопасно
3. Cloudflare Tunnel создает публичный URL - используйте только для тестирования
4. В production используйте нормальный домен и HTTPS

## Остановка

```bash
docker compose down
```

Это остановит все сервисы, но **НЕ удалит volumes** (данные сохранятся).

Для полной очистки (включая данные):
```bash
docker compose down -v
```

⚠️ **Внимание:** Это удалит все данные базы данных!

## Мониторинг

### Проверка здоровья API

```bash
curl http://localhost/api/health
```

### Проверка UI

Откройте в браузере публичный URL от cloudflared или локально:
```bash
# Локально (если открыт порт 80)
curl http://localhost/

# Через cloudflared URL
curl https://xxx.trycloudflare.com/
```

### Статистика базы данных

Подключитесь к базе:
```bash
docker compose exec db psql -U raft_user -d raft
```

Запросы:
```sql
-- Количество чанков
SELECT count(*) FROM chunks;

-- Количество секций
SELECT count(*) FROM sections;

-- Чанки с эмбеддингами
SELECT count(*) FROM chunks WHERE embedding IS NOT NULL;
```

## Обновление

### Обновление кода из GitHub

```bash
docker compose restart repo-sync api ui
```

### Обновление Docker образов

```bash
docker compose build --no-cache
docker compose up -d
```

### Обновление базы данных

Если нужно обновить дамп:

1. Создайте новый дамп на локальной машине
2. Скопируйте на сервер
3. Пересоздайте базу:

```bash
docker compose down
docker volume rm raft_dbinit_state raft_pgdata
# Замените raft.dump на новый файл
docker compose up -d
```
