FROM python:3.11-slim

LABEL maintainer="Ahmed Hassan <a.hassan.b.h@gmail.com>"
LABEL description="Hermes-Lite — cloud-first AI agent via NVIDIA NIM Free API"
LABEL version="0.6.0"

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY hermes_lite/ ./hermes_lite/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Install the package
RUN pip install --no-cache-dir -e ".[test]"

# Create data directory for SQLite + logs
RUN mkdir -p /root/.hermes_lite/logs

# Default environment
ENV HERMES_LITE_CLOUD_URL=https://integrate.api.nvidia.com/v1
ENV HERMES_LITE_CLOUD_MODEL=z-ai/glm-5.2
ENV HERMES_LITE_LOCAL_URL=http://127.0.0.1:8080/v1
ENV HERMES_LITE_LOCAL_MODEL=Qwen2.5-Coder-7B-Instruct-IQ3_XS.gguf

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "--tb=short", "-q"]
