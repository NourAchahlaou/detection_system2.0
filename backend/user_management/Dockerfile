FROM python:3.12

# set work directory
WORKDIR /usr/srv

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# create non-root user and set working directory
RUN useradd -rm -d /code -s /bin/bash -u 1001 ubuntu

# Install system dependencies
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

# copy requirements file
COPY ./user_management/requirements.txt /usr/srv/requirements.txt

# install requirements
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copy project files
COPY ./user_management /usr/srv/user_management

# make sure migrations directory exists
RUN mkdir -p /usr/srv/user_management/migrations

RUN chmod +x /usr/srv/user_management/entrypoint.sh

# give ubuntu user ownership of the project files
RUN chown -R ubuntu:ubuntu /usr/srv

# switch to non-root user
USER ubuntu

# expose port
EXPOSE 8000

# use the entrypoint script to run migrations before starting the app
ENTRYPOINT ["/usr/srv/user_management/entrypoint.sh"]
