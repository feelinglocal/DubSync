FROM node:24-bookworm-slim AS frontend

WORKDIR /frontend
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DUBSYNC_DATA_DIR=/var/data \
    DUBSYNC_PROVIDERS_PATH=/app/provider.yaml \
    DUBSYNC_STYLE_PATH=/app/style_profile.yaml \
    DUBSYNC_STATIC_DIR=/app/web/dist

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 dubsync \
    && mkdir -p /app /var/data \
    && chown -R dubsync:dubsync /app /var/data

WORKDIR /app
COPY --chown=dubsync:dubsync pyproject.toml README.md ./
COPY --chown=dubsync:dubsync src/ ./src/
COPY --chown=dubsync:dubsync provider.yaml style_profile.yaml ./
COPY --from=frontend --chown=dubsync:dubsync /frontend/dist ./web/dist

RUN python -m pip install --upgrade "pip>=26.1.2" \
    && python -m pip install ".[cloud,web]"

USER dubsync
EXPOSE 10000

CMD ["dubsync-web"]
