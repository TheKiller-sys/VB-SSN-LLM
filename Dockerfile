FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Instalar PyTorch CPU-only (índice dedicado, ~200 MB en vez de ~2.5 GB)
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1

# 2) Resto de dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Código
COPY . .

# Usuario no-root (opcional pero recomendado)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
