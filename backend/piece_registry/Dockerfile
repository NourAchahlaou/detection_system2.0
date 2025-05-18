FROM python:3.12

WORKDIR /usr/srv

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN useradd -rm -d /code -s /bin/bash -u 1001 ubuntu

# Base dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    ca-certificates \
    libopencv-dev \
    libusb-1.0-0 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    v4l-utils \
    usbutils \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY ./pylon/pylon_8.1.0-deb0_amd64.deb /tmp/pylon.deb
RUN apt-get update && \
    apt-get install -y /tmp/pylon.deb && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /tmp/pylon.deb

# Set Pylon environment variables
ENV PYLON_ROOT=/opt/pylon
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYLON_ROOT}/lib64

COPY ./requirements.txt /usr/srv/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
# Set Python path to include application directory
ENV PYTHONPATH="${PYTHONPATH}:/usr/srv/piece_registry:/usr/srv"
COPY . /usr/srv/piece_registry
RUN mkdir -p /usr/srv/piece_registry/migrations
RUN chmod 755 /usr/srv/piece_registry/entrypoint.sh
RUN chown -R ubuntu:ubuntu /usr/srv

# Verify that the entrypoint file exists and has correct permissions
RUN ls -la /usr/srv/piece_registry/entrypoint.sh

USER ubuntu

EXPOSE 8000

ENTRYPOINT ["/usr/srv/piece_registry/entrypoint.sh"]