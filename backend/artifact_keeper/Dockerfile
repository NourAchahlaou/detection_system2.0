FROM python:3.12

WORKDIR /usr/srv

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN useradd -rm -d /code -s /bin/bash -u 1001 ubuntu

RUN apt-get update && apt-get install -y postgresql-client ca-certificates && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /usr/srv/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /usr/srv/artifact_keeper
RUN mkdir -p /usr/srv/artifact_keeper/migrations
RUN chmod 755 /usr/srv/artifact_keeper/entrypoint.sh
RUN chown -R ubuntu:ubuntu /usr/srv

# Verify that the entrypoint file exists and has correct permissions
RUN ls -la /usr/srv/artifact_keeper/entrypoint.sh

USER ubuntu

EXPOSE 8000

ENTRYPOINT ["/usr/srv/artifact_keeper/entrypoint.sh"]