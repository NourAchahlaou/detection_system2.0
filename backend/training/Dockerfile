FROM airvision_base:latest

WORKDIR /usr/srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1


# Install system dependencies including OpenGL libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
# # Create user and directories with proper permissions
# RUN useradd -rm -d /code -s /bin/bash -u 1001 ubuntu
# Copy application code
COPY . /usr/srv/training

RUN mkdir -p /usr/srv/training/migrations

# # Install Python dependencies
# COPY ./requirements.txt /usr/srv/requirements.txt
# RUN pip install --no-cache-dir --upgrade -r requirements.txt



RUN chmod +x /usr/srv/training/entrypoint.sh

# Verify permissions
RUN ls -la /usr/srv/training/entrypoint.sh

# Create shared data directories with proper permissions for ubuntu user
RUN mkdir -p /app/shared/dataset /app/shared/models /app/shared/logs /app/shared/backup
RUN chown -R ubuntu:ubuntu /app/shared
RUN chmod -R 755 /app/shared

# Also create the local shared_data directories with proper permissions
RUN mkdir -p ./shared_data/logs ./shared_data/models
RUN chown -R ubuntu:ubuntu ./shared_data
RUN chmod -R 755 ./shared_data

# Create a cache directory for YOLO models that the ubuntu user can write to
RUN mkdir -p /home/ubuntu/.cache/yolo
RUN chown -R ubuntu:ubuntu /home/ubuntu/.cache
RUN chmod -R 755 /home/ubuntu/.cache

# Switch to non-root user
USER ubuntu

EXPOSE 8000

ENTRYPOINT ["/usr/srv/training/entrypoint.sh"]