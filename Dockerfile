# syntax=docker/dockerfile:1
# AIS Collision Detection Pipeline
# Requires Java for PySpark and system geo libraries for GeoPandas.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Java (required by PySpark) and geo system libraries (required by GeoPandas).
# default-jre-headless works on both amd64 and arm64 (Apple Silicon).
RUN apt-get update && apt-get install -y \
    default-jre-headless \
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Java home dynamically — resolves correctly on both amd64 and arm64.
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]