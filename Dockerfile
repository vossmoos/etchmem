FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Install dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# To use the offline local embedding backend instead of OpenAI, uncomment:
# COPY requirements-local.txt .
# RUN pip install --no-cache-dir -r requirements-local.txt

COPY app ./app

# Declarative extension vocabulary (ext/*.yaml). Baked in by default; mount a
# volume over /srv/ext (or set ETCHMEM_EXT_DIR) to edit without rebuilding.
COPY ext ./ext
ENV ETCHMEM_EXT_DIR=/srv/ext

# Data dir for the two DuckDB files (mount a volume here in compose).
ENV ETCHMEM_DATA_DIR=/data
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
