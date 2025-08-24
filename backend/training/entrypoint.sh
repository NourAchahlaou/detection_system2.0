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
DB_PORT=${POSTGRES_PORT:-5432}
DB_PASSWORD=${POSTGRES_PASSWORD}

# Print connection details (excluding password)
echo "Database connection details:"
echo "  Host: $DB_HOST"
echo "  Port: $DB_PORT"
echo "  User: $DB_USER"
echo "  Database: $DB_NAME"
echo "  Password: [REDACTED]"

# Wait for basic database connectivity
while [ $RETRIES -lt $MAX_RETRIES ]; do
    echo "Attempt $RETRIES: Checking if database is ready..."
    
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

echo "Database is ready, performing comprehensive checks..."

# Create schemas directly using psql
echo "Creating schemas in database..."
if [ -z "$DB_PASSWORD" ]; then
    echo "WARNING: DB_PASSWORD is not set or empty"
fi

export PGPASSWORD="$DB_PASSWORD"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "CREATE SCHEMA IF NOT EXISTS training;"

echo "Schemas created successfully!"

# Run Alembic migrations
echo "Running Alembic migrations..."
cd training
alembic upgrade head
if [ $? -eq 0 ]; then
    echo "Alembic migrations completed successfully!"
else
    echo "ERROR: Alembic migrations failed!"
    exit 1
fi
cd ..

# Verify database setup (do this once at startup)
echo "Verifying database setup..."
VERIFICATION_RESULT=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
    SELECT COUNT(*) 
    FROM information_schema.tables 
    WHERE table_schema = 'training' 
")

EXPECTED_TABLES=1

ACTUAL_TABLES=$(echo $VERIFICATION_RESULT | tr -d ' ')

if [ "$ACTUAL_TABLES" -eq "$EXPECTED_TABLES" ]; then
    echo "✓ All expected tables are present ($ACTUAL_TABLES/$EXPECTED_TABLES)"
else
    echo "⚠ Warning: Expected $EXPECTED_TABLES tables, found $ACTUAL_TABLES"
    # List available tables for debugging
    echo "Available tables in training schema:"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'training'
        ORDER BY table_name;
    "
fi

echo "Database setup verification complete!"

# ========================================
# YOLO MODEL DOWNLOAD SECTION
# ========================================

echo "Setting up YOLO model..."

# Set model paths based on environment variables or defaults
MODELS_BASE_PATH=${MODELS_BASE_PATH:-/app/shared/models}
YOLO_MODEL_PATH="$MODELS_BASE_PATH/yolov8n.pt"
YOLO_MODEL_URL="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

echo "Model configuration:"
echo "  Models base path: $MODELS_BASE_PATH"
echo "  YOLO model path: $YOLO_MODEL_PATH"
echo "  Download URL: $YOLO_MODEL_URL"

# Create models directory if it doesn't exist
echo "Creating models directory..."
mkdir -p "$MODELS_BASE_PATH"

# Check if model already exists
if [ -f "$YOLO_MODEL_PATH" ]; then
    echo "✓ YOLO model already exists at $YOLO_MODEL_PATH"
    
fi

# Download model if it doesn't exist or was removed
if [ ! -f "$YOLO_MODEL_PATH" ]; then
    echo "Downloading YOLO model..."
    echo "This may take a few minutes depending on your internet connection..."
    
    # Use curl with progress bar and retry options
    if curl -L --progress-bar --retry 3 --retry-delay 2 -o "$YOLO_MODEL_PATH" "$YOLO_MODEL_URL"; then
        echo "✓ YOLO model downloaded successfully!"
        
        # Verify the downloaded file
        if [ -f "$YOLO_MODEL_PATH" ]; then
            DOWNLOADED_SIZE=$(stat -c%s "$YOLO_MODEL_PATH")
            echo "✓ Downloaded file size: $DOWNLOADED_SIZE bytes"
            
        else
            echo "ERROR: Model file was not created after download"
            exit 1
        fi
    else
        echo "ERROR: Failed to download YOLO model"
        exit 1
    fi
fi

# Final verification
if [ -f "$YOLO_MODEL_PATH" ]; then
    echo "✓ YOLO model setup complete!"
    ls -lh "$YOLO_MODEL_PATH"
else
    echo "ERROR: YOLO model setup failed"
    exit 1
fi

echo "Model setup verification complete!"

# ========================================
# END YOLO MODEL DOWNLOAD SECTION
# ========================================

# Start the application
echo "Starting application"
exec uvicorn training.app.main:app --host 0.0.0.0 --port 8000 --reload