# Builder stage
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH="/opt/venv"

# Build зависимости (нужны только на этапе сборки)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# виртуальное окружение под зависимости
RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /app

# сначала зависимости чтобы лучше работал docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt


# runtime stage минимальный рантайм: только venv и код
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH="/app/src" \
    VENV_PATH="/opt/venv" \
    PATH="/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# runtime-зависимости (нужны для xgboost / openmp)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# запускается под non-root пользователем
RUN groupadd -g 10001 tsfit \
    && useradd -m -u 10001 -g 10001 -s /usr/sbin/nologin tsfit

WORKDIR /app

# готовое виртуальное окружение из builder
COPY --from=builder "${VENV_PATH}" "${VENV_PATH}"

# копировать только нужный код (который src)
COPY --chown=tsfit:tsfit src/ /app/src/

# каталог под MLflow (db + artifacts)
RUN mkdir -p /mlflow/artifacts \
    && chown -R tsfit:tsfit /mlflow

VOLUME ["/mlflow"]

EXPOSE 8000

USER tsfit

# По умолчанию — API. В docker-compose можно переопределить команду для mlflow server.
CMD ["uvicorn", "tsfit.main:app", "--host", "0.0.0.0", "--port", "8000"]