#!/bin/bash
set -e

BACKUP_FILE="${1:-/backup/raft.dump}"
STATE_DIR="${2:-/state}"
DB_NAME="${POSTGRES_DB:-raft}"
# Use superuser for DROP/CREATE DATABASE operations
DB_SUPERUSER="${POSTGRES_SUPERUSER:-postgres}"
DB_USER="${POSTGRES_USER:-raft_user}"
DB_HOST="${POSTGRES_HOST:-db}"
DB_PORT="${POSTGRES_PORT:-5432}"

MARKER_FILE="$STATE_DIR/RESTORED_OK"

echo "Database restore script"
echo "BACKUP_FILE: $BACKUP_FILE"
echo "DB_NAME: $DB_NAME"
echo "DB_SUPERUSER: $DB_SUPERUSER"
echo "DB_USER: $DB_USER"
echo "DB_HOST: $DB_HOST"
echo "DB_PORT: $DB_PORT"

# Check if marker file exists (already restored)
if [ -f "$MARKER_FILE" ]; then
    echo "Database already restored (marker file exists: $MARKER_FILE)"
    echo "Skipping restore..."
    exit 0
fi

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "Waiting for PostgreSQL to be ready..."

# Wait for PostgreSQL to be ready (using superuser)
until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_SUPERUSER" > /dev/null 2>&1; do
    echo "PostgreSQL is not ready, waiting..."
    sleep 2
done

echo "PostgreSQL is ready"

# Create state directory if it doesn't exist
mkdir -p "$STATE_DIR"

# Check if database exists and drop if needed (for clean restore)
echo "Checking if database exists..."
DB_EXISTS=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_SUPERUSER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1 && echo "yes" || echo "no")

if [ "$DB_EXISTS" = "yes" ]; then
    echo "Database exists, dropping it for clean restore..."
    # Terminate active connections before DROP
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_SUPERUSER" -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='$DB_NAME' AND pid <> pg_backend_pid();" || true
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_SUPERUSER" -d postgres -c "DROP DATABASE $DB_NAME;"
fi

# Create database (using superuser)
echo "Creating database..."
PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_SUPERUSER" -d postgres -c "CREATE DATABASE $DB_NAME;"

# Enable pgvector extension (using superuser)
echo "Enabling pgvector extension..."
PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_SUPERUSER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"
PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_SUPERUSER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"

# Restore database from dump (using superuser, but objects will be owned by DB_USER after restore)
echo "Restoring database from dump..."
PGPASSWORD="$POSTGRES_PASSWORD" pg_restore \
    --no-owner \
    --no-privileges \
    --verbose \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_SUPERUSER" \
    -d "$DB_NAME" \
    "$BACKUP_FILE"

# Verify restore
echo "Verifying restore..."
TABLE_COUNT=$(PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_SUPERUSER" -d "$DB_NAME" -tAc "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';")
echo "Tables restored: $TABLE_COUNT"

if [ "$TABLE_COUNT" -eq 0 ]; then
    echo "WARNING: No tables found after restore"
else
    # Create marker file
    echo "$(date)" > "$MARKER_FILE"
    echo "Database restore completed successfully"
    echo "Marker file created: $MARKER_FILE"
fi
