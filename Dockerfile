FROM langchain/langgraph-api:3.13


WORKDIR /app
COPY . .

# Force CPU-only torch BEFORE installing everything else
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install langgraph-cli
RUN pip install -e ".[dev]" || pip install -e .

EXPOSE 2024