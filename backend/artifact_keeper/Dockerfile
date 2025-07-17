FROM airvision_base:latest

WORKDIR /usr/srv




# # Create user and directories with proper permissions
# RUN useradd -rm -d /code -s /bin/bash -u 1001 ubuntu

# Create all necessary directories as root first
RUN mkdir -p /app/shared/dataset/piece/piece
RUN mkdir -p /usr/srv/artifact_keeper/migrations

# # Install Python dependencies
# COPY ./requirements.txt /usr/srv/requirements.txt
# RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY . /usr/srv/artifact_keeper

# Set proper ownership and permissions
RUN chown -R ubuntu:ubuntu /usr/srv /app/shared
RUN chmod -R 755 /app/shared /usr/srv/artifact_keeper
RUN chmod +x /usr/srv/artifact_keeper/entrypoint.sh

# Verify permissions
RUN ls -la /usr/srv/artifact_keeper/entrypoint.sh
RUN ls -la /app/shared/dataset/

# Switch to non-root user
USER ubuntu

EXPOSE 8000

ENTRYPOINT ["/usr/srv/artifact_keeper/entrypoint.sh"]