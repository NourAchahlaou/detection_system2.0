#!/bin/bash
set -e

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | tr -d '\r' | xargs -0)
    echo "Loaded environment variables from .env file"
else
    echo "No .env file found, using existing environment variables"
fi


# Start the application
echo "Starting application"
exec uvicorn video_streaming.app.main:app --host 0.0.0.0 --port 8000 --reload