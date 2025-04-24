FROM python:3.12

# set work directory
WORKDIR /usr/srv

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# create non-root user and set working directory
RUN useradd -rm -d /code -s /bin/bash -u 1001 ubuntu

# copy requirements file
COPY ./requirements.txt /usr/srv/requirements.txt

# install requirements
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# switch to non-root user
USER ubuntu

# expose port
EXPOSE 8000

# run the FastAPI app
CMD bash -c 'uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload'
