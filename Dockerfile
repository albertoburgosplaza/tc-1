
# Dockerfile único con perfiles de instalación por servicio
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# SERVICE puede ser: app | ingest | pyexec
ARG SERVICE=app

COPY requirements.app.txt requirements.ingest.txt requirements.pyexec.txt /tmp/

RUN pip install --no-cache-dir -r /tmp/requirements.${SERVICE}.txt

COPY . /app

# Crear usuario no-root para pyexec por seguridad
RUN if [ "$SERVICE" = "pyexec" ] ; then \
    groupadd -r pyexec && \
    useradd -r -g pyexec -d /app -s /bin/bash pyexec && \
    chown -R pyexec:pyexec /app ; \
    fi

# Expose all ports that services might use
EXPOSE 7860 8001

# El comando se define desde docker-compose por servicio
