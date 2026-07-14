FROM python:3.12-slim
ENV TF_USE_LEGACY_KERAS=1
WORKDIR /app

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps tensorflow-recommenders==0.7.3

# refresh_index.py needs the aiplatform SDK — which will yank protobuf to 6.x and
# break TensorFlow inside this image, exactly like it did in your venv. Re-assert
# the pins immediately afterwards. Order matters; do not reorder these two lines.
RUN pip install --no-cache-dir google-cloud-aiplatform
RUN pip install --no-cache-dir "protobuf==4.25.5" "numpy==1.26.4"

# Fail the BUILD, not the pipeline at 3am, if the pins didn't survive.
RUN python -c "import numpy, google.protobuf, tensorflow as tf, tensorflow_recommenders; \
import google.cloud.aiplatform; \
print('deps ok:', numpy.__version__, google.protobuf.__version__, tf.__version__)"

COPY training/train.py .
COPY featurestore/refresh_index.py .
COPY pipelines/evaluate.py .
COPY pipelines/hold.py .