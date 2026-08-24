FROM python:3.14-slim@sha256:ce40764625a4ff50df3548277632e7f96c4e77fe75fa848aae9885476e7df5a4
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
