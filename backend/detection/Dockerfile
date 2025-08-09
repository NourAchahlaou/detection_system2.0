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
COPY ./requirements.txt /usr/srv/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
# Copy application code
COPY . /usr/srv/detection

# Make entrypoint executable
RUN chmod +x /usr/srv/detection/entrypoint.sh

# Verify permissions
RUN ls -la /usr/srv/detection/entrypoint.sh

# Create only the models shared directory (since you don't need dataset)
RUN mkdir -p /app/shared/models
RUN chown -R ubuntu:ubuntu /app/shared
RUN chmod -R 755 /app/shared

# Create local shared_data directory only for models
RUN mkdir -p ./shared_data/models
RUN chown -R ubuntu:ubuntu ./shared_data
RUN chmod -R 755 ./shared_data

# Create a cache directory for YOLO models that the ubuntu user can write to
RUN mkdir -p /home/ubuntu/.cache/yolo
RUN chown -R ubuntu:ubuntu /home/ubuntu/.cache
RUN chmod -R 755 /home/ubuntu/.cache

# Switch to non-root user
USER ubuntu

EXPOSE 8000

ENTRYPOINT ["/usr/srv/detection/entrypoint.sh"]