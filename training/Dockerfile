FROM python:3.15-slim@sha256:8c13a3f37a6a5b71498228e902c4b9f51c1f7900d49fabf71f9a650c6120e0d0
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
