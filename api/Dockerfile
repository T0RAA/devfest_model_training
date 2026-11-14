FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8080

COPY pyproject.toml README.md SPEC.md MILESTONES_v2_detector_classifier.md ./
COPY configs/ ./configs/
COPY data/splits/classes.json ./data/splits/classes.json
COPY src/ ./src/
COPY api/ ./api/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[ml,cloud]" \
    "PytorchWildlife==1.2.0.5" \
    "open_clip_torch==2.26.1" \
    "timm==1.0.9"

EXPOSE 8080

# Designed for Cloud Run 2 vCPU / 4 GiB instance
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
