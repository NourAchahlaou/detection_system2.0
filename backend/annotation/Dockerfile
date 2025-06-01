FROM python:3.12

WORKDIR /usr/srv

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y postgresql-client ca-certificates && rm -rf /var/lib/apt/lists/*

# Create user and directories with proper permissions
RUN useradd -rm -d /code -s /bin/bash -u 1001 ubuntu

# Create all necessary directories as root first
RUN mkdir -p /app/shared/dataset/piece/piece
RUN mkdir -p /usr/srv/annotation/migrations

# Install Python dependencies
COPY ./requirements.txt /usr/srv/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY . /usr/srv/annotation

# Set proper ownership and permissions
RUN chown -R ubuntu:ubuntu /usr/srv /app/shared
RUN chmod -R 755 /app/shared /usr/srv/annotation
RUN chmod +x /usr/srv/annotation/entrypoint.sh

# Verify permissions
RUN ls -la /usr/srv/annotation/entrypoint.sh
RUN ls -la /app/shared/dataset/

# Switch to non-root user
USER ubuntu

EXPOSE 8000

ENTRYPOINT ["/usr/srv/annotation/entrypoint.sh"]