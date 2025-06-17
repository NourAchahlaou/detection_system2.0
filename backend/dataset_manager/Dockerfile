FROM python:3.12

WORKDIR /usr/srv

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y postgresql-client ca-certificates && rm -rf /var/lib/apt/lists/*

# Create user and directories with proper permissions
RUN useradd -rm -d /code -s /bin/bash -u 1001 ubuntu


RUN mkdir -p /usr/srv/dataset_manager/migrations

# Install Python dependencies
COPY ./requirements.txt /usr/srv/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt


RUN chmod +x /usr/srv/dataset_manager/entrypoint.sh

# Verify permissions
RUN ls -la /usr/srv/dataset_manager/entrypoint.sh

# Switch to non-root user
USER ubuntu

EXPOSE 8000

ENTRYPOINT ["/usr/srv/dataset_manager/entrypoint.sh"]