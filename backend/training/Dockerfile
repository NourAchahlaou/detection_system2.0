FROM python:3.12

WORKDIR /usr/srv

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies including OpenGL libraries
RUN apt-get update && apt-get install -y \
    postgresql-client \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create user and directories with proper permissions
RUN useradd -rm -d /code -s /bin/bash -u 1001 ubuntu

RUN mkdir -p /usr/srv/training/migrations

# Install Python dependencies
COPY ./requirements.txt /usr/srv/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY . /usr/srv/training 

RUN chmod +x /usr/srv/training/entrypoint.sh

# Verify permissions
RUN ls -la /usr/srv/training/entrypoint.sh

# Create shared data directories with proper permissions
RUN mkdir -p ./shared_data/logs ./shared_data/models
RUN chmod 755 ./shared_data/logs ./shared_data/models

# Switch to non-root user
USER ubuntu

EXPOSE 8000

ENTRYPOINT ["/usr/srv/training/entrypoint.sh"]