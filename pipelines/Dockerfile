FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a
ENV TF_USE_LEGACY_KERAS=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps tensorflow-recommenders==0.7.3

# Keep the SDK and TensorFlow on the explicitly tested NumPy/protobuf pair.
# Re-assert the pins immediately afterwards. Order matters; do not reorder.
RUN pip install --no-cache-dir "google-cloud-aiplatform==1.165.1"
RUN pip install --no-cache-dir "protobuf==6.33.6" "numpy==1.26.4"

# Fail the BUILD, not the pipeline at 3am, if the pins didn't survive.
RUN python -c "import numpy, google.protobuf, tensorflow as tf, tensorflow_recommenders; \
import google.cloud.aiplatform; \
print('deps ok:', numpy.__version__, google.protobuf.__version__, tf.__version__)"

COPY training/train.py .
COPY training/generation_contract.py .
COPY featurestore/refresh_index.py .
COPY featurestore/gc_generation.py .
COPY pipelines/evaluate.py .
COPY pipelines/promote.py .
COPY pipelines/hold.py .

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid 10001 --home-dir /tmp --no-create-home app \
    && chown -R 10001:10001 /app
USER 10001:10001
