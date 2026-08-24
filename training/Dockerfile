FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a
ENV TF_USE_LEGACY_KERAS=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Same two-step install as your laptop: TFRS with --no-deps, so it can't drag a second
# TensorFlow in. Keeping the container and the venv byte-identical is the whole point.
RUN pip install --no-cache-dir --no-deps tensorflow-recommenders==0.7.3
COPY train.py .
COPY generation_contract.py .
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --home-dir /tmp --no-create-home app \
    && chown -R 10001:10001 /app
USER 10001:10001
ENTRYPOINT ["python", "train.py"]
