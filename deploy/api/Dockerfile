FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install minimal runtime dependencies (fallback if uv sync fails)
# Actual dependencies will be installed from /repo at container startup
RUN uv pip install --system \
    fastapi \
    uvicorn[standard] \
    psycopg[binary] \
    || echo "Warning: Failed to install minimal deps, will use uv sync from /repo"

# Set working directory (code will be in /repo volume)
WORKDIR /repo

# PATH will be set to /repo/.venv/bin in docker-compose command
# Default command (will be overridden in docker-compose.yml)
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
