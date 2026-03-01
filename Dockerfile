FROM python:3.13

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY . .

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e ".[dev]" || pip install -e .

# If you want to keep executing it directly (optional if you use bash CMD)
RUN chmod +x ./scripts/run_aegra.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ok || exit 1

EXPOSE 8000

# Most robust:
CMD ["bash", "./scripts/run_aegra.sh", "serve"]