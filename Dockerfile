FROM bitnami/pytorch:1.13.1-debian-11-r25

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/images /app/models/transformers
VOLUME /app/images
VOLUME /app/models

ENV EAGER_LOAD=1 \
    MODEL_PATH=/app/models/transformers \
    PYTHONUNBUFFERED=1

EXPOSE 5000

HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/healthz', timeout=3)" || exit 1

CMD ["python3", "app.py"]
