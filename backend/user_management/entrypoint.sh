#!/bin/bash
set -e

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs -d '\n')

echo "Starting entrypoint script"

# Wait for database to be ready
echo "Waiting for database..."
MAX_RETRIES=30
RETRY_INTERVAL=2
RETRIES=0

DB_HOST=${POSTGRES_HOST}
DB_USER=${POSTGRES_USER}
DB_NAME=${POSTGRES_DB}

while [ $RETRIES -lt $MAX_RETRIES ]; do
    echo "Attempt $RETRIES: Checking if database is ready..."

    if pg_isready -h $DB_HOST -U $DB_USER -d $DB_NAME; then
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

# Go to the directory where alembic.ini exists
cd /usr/srv/user_management

# Only apply existing migrations
alembic upgrade head || echo "Migration failed, but continuing startup"


cd ..
# Start the application
echo "Starting application"
exec uvicorn user_management.app.main:app --host 0.0.0.0 --port 8000 --reload
