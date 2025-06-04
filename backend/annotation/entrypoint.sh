#!/bin/bash
set -e

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | tr -d '\r' | xargs -0)
    echo "Loaded environment variables from .env file"
else
    echo "No .env file found, using existing environment variables"
fi

echo "Starting entrypoint script"

# Wait for database to be ready
echo "Waiting for database..."
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRIES=0

# Validate required environment variables
DB_HOST=${POSTGRES_HOST:-databasePostgres}
DB_USER=${POSTGRES_USER:-airbususer}
DB_NAME=${POSTGRES_DB:-airbusdb}
DB_PORT=${POSTGRES_PORT:-5432}  # Default to 5432 if not specified
DB_PASSWORD=${POSTGRES_PASSWORD}

# Print connection details (excluding password)
echo "Database connection details:"
echo "  Host: $DB_HOST"
echo "  Port: $DB_PORT"
echo "  User: $DB_USER"
echo "  Database: $DB_NAME"
echo "  Password: [REDACTED]"

while [ $RETRIES -lt $MAX_RETRIES ]; do
    echo "Attempt $RETRIES: Checking if database is ready..."
    
    # Use PGPASSWORD environment variable for pg_isready
    export PGPASSWORD="$DB_PASSWORD"
    if pg_isready -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -p "$DB_PORT"; then
        echo "Database is ready!"
        break
    fi
    
    RETRIES=$((RETRIES+1))
    echo "Database not ready yet (attempt $RETRIES/$MAX_RETRIES)... waiting"
    sleep $RETRY_INTERVAL
done

if [ $RETRIES -eq $MAX_RETRIES ]; then
    echo "Could not connect to database, giving up"
    exit 1
fi

echo "Database is ready, running migrations"

# Create schemas directly using psql
echo "Creating schemas in database..."
# Make sure DB_PORT is set and use it correctly and ensure password is passed correctly
if [ -z "$DB_PASSWORD" ]; then
    echo "WARNING: DB_PASSWORD is not set or empty"
fi

# Use PGPASSWORD environment variable and make sure it's exported
export PGPASSWORD="$DB_PASSWORD"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "CREATE SCHEMA IF NOT EXISTS annotation;"

echo "Schemas created successfully!"


# Run Alembic migrations
cd annotation
alembic upgrade head
if [ $? -eq 0 ]; then
    echo "Alembic migrations completed successfully!"
else
    echo "ERROR: Alembic migrations failed!"
    exit 1
fi
cd .. 
# Start the application
echo "Starting application"
exec uvicorn annotation.app.main:app --host 0.0.0.0 --port 8000 --reload