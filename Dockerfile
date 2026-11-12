FROM python:3.11-slim

# Evita gravacao de bytecode (.pyc) e garante logs em tempo real no container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Instala libgomp1 exigida pelo motor de inferencia paralela do LightGBM em Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instala dependencias Python aproveitando cache de camadas do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o codigo-fonte, politicas, modelo calibrado e warehouse processado
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY data/policies/ ./data/policies/
COPY data/processed/ ./data/processed/

# Cria usuario nao-privilegiado (non-root) por seguranca de producao
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://127.0.0.1:8000/health'); r.raise_for_status()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
