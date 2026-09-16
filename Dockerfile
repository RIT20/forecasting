FROM  python:3.9-slim-buster
LABEL maintainer="sudhakar.g@hypersonix.ai"

# set working directory
WORKDIR /app

RUN mkdir -p /var/model_save
RUN mkdir -p /var/logs

RUN apt-get update \
    && apt-get install curl -y \
    && apt-get install gcc -y \
    && apt-get clean

# Install and upgrade pip
RUN pip install --upgrade pip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy using poetry.lock* in case it doesn't exist yet
COPY ./pyproject.toml ./poetry.lock* /app/

RUN poetry install --no-root --no-dev

# copy project
COPY . /app/

ENTRYPOINT ["python","dag_scripts/dag_runner.py"]



