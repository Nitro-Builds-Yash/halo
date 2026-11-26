# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set RERANK=0 to build a smaller image without torch / the cross-encoder.
ARG RERANK=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CHROMA_PERSIST_DIR=/data/chroma_db \
    HF_HOME=/data/hf_cache

WORKDIR /app

COPY pyproject.toml README.md main.py ./
COPY src ./src
RUN if [ "$RERANK" = "1" ]; then \
        pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[ui,rerank]"; \
    else \
        pip install -e ".[ui]"; \
    fi

RUN useradd --create-home --uid 1000 halo && mkdir -p /data && chown -R halo:halo /data /app
USER halo
VOLUME ["/data"]

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '8000'))" || exit 1

# Listens on $PORT when the platform sets it (Render, Railway, Cloud Run), else 8000.
CMD ["halo", "serve", "--host", "0.0.0.0"]
