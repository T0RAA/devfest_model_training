FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8080

COPY pyproject.toml README.md SPEC.md ./
COPY configs/ ./configs/
COPY data/splits/classes.json ./data/splits/classes.json
COPY src/ ./src/
COPY api/ ./api/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[ml,cloud]"

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
