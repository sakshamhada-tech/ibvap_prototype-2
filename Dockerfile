FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.lock ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.lock

COPY . .
RUN useradd --create-home --uid 10001 ibvap \
    && mkdir -p /app/logs /app/output \
    && chown -R ibvap:ibvap /app/logs /app/output

USER ibvap
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"

CMD ["python", "server.py"]
