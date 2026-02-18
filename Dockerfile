FROM langchain/langgraph-api:3.13

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install the package and all dependencies
RUN pip install -e ".[dev]" || pip install -e .

EXPOSE 2024