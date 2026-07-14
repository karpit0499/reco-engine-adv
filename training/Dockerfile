FROM python:3.12-slim
ENV TF_USE_LEGACY_KERAS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Same two-step install as your laptop: TFRS with --no-deps, so it can't drag a second
# TensorFlow in. Keeping the container and the venv byte-identical is the whole point.
RUN pip install --no-cache-dir --no-deps tensorflow-recommenders==0.7.3
COPY train.py .
ENTRYPOINT ["python", "train.py"]