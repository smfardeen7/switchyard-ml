FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SWITCHYARD_ENV=production \
    SWITCHYARD_DATA_DIR=/app/state/data \
    SWITCHYARD_MODEL_DIR=/app/state/models
WORKDIR /app
RUN groupadd --gid 10001 switchyard && useradd --uid 10001 --gid switchyard --no-create-home switchyard
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=10001:10001 switchyard ./switchyard
RUN mkdir -p /app/state/data /app/state/models && chown -R 10001:10001 /app/state
USER 10001:10001
EXPOSE 8090
HEALTHCHECK --interval=10s --timeout=3s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/ready',timeout=2)"
CMD ["uvicorn", "switchyard.api:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
