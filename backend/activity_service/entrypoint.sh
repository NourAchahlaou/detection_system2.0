#!/bin/bash

set -e

echo "Starting Activity Service..."

# Wait for database to be ready
echo "Waiting for PostgreSQL to be ready..."
python -c "
import time
import psycopg2
import os
from urllib.parse import urlparse

# Parse database URL from environment
db_url = os.getenv('DATABASE_URL')
if db_url:
    parsed = urlparse(db_url)
    host = parsed.hostname
    port = parsed.port or 5432
    user = parsed.username
    password = parsed.password
    database = parsed.path[1:]  # Remove leading slash
else:
    # Fallback to individual environment variables
    host = os.getenv('POSTGRES_HOST', 'databasePostgres')
    port = int(os.getenv('POSTGRES_PORT', '5432'))
    user = os.getenv('POSTGRES_USER', 'airbususer')
    password = os.getenv('POSTGRES_PASSWORD', 'airbuspassword')
    database = os.getenv('POSTGRES_DB', 'airvisiondb')

max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        conn.close()
        print('PostgreSQL is ready!')
        break
    except psycopg2.OperationalError:
        retry_count += 1
        print(f'PostgreSQL is not ready yet. Retrying... ({retry_count}/{max_retries})')
        time.sleep(2)
else:
    print('Failed to connect to PostgreSQL after maximum retries')
    exit(1)
"

# Wait for RabbitMQ to be ready
echo "Waiting for RabbitMQ to be ready..."
python -c "
import time
import pika
import os

rabbitmq_url = os.getenv('RABBITMQ_URL', 'amqp://airvision:airvision123@airvision_rabbitmq:5672/airvision')
max_retries = 30
retry_count = 0

while retry_count < max_retries:
    try:
        connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
        connection.close()
        print('RabbitMQ is ready!')
        break
    except Exception as e:
        retry_count += 1
        print(f'RabbitMQ is not ready yet. Retrying... ({retry_count}/{max_retries})')
        time.sleep(2)
else:
    print('Failed to connect to RabbitMQ after maximum retries')
    exit(1)
"

# Navigate to the application directory
cd /usr/srv/activity_service

# Run database migrations if needed
echo "Running database migrations..."
python -m alembic upgrade head || echo "No migrations to run or alembic not configured yet"

# Start the FastAPI application
echo "Starting Activity Service on port 8000..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload