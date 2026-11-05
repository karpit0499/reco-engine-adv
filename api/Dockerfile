FROM python:3.15-slim@sha256:8c13a3f37a6a5b71498228e902c4b9f51c1f7900d49fabf71f9a650c6120e0d0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache

RUN addgroup --system --gid 10001 app && \
    adduser --system --uid 10001 --ingroup app --home /tmp --no-create-home app

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=10001:10001 . .

USER 10001:10001
CMD exec gunicorn --bind :${PORT:-8080} --worker-tmp-dir /tmp --workers 1 --threads 8 main:app
