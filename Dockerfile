FROM python:3.13

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY . .

# Force CPU-only torch BEFORE installing everything else.
RUN pip install --upgrade pip
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e ".[dev]" || pip install -e .

EXPOSE 2024
CMD ["./scripts/run_aegra.sh", "serve"]
