# TruthLens on a Hugging Face Space (Docker SDK, free "CPU basic" hardware).
# Same Python and package versions as the tested local setup; models are baked in at build time so a visitor
# never waits for downloads.
FROM python:3.15-slim
LABEL org.opencontainers.image.source=https://github.com/Yashas95/yuvaHacks

# retried: Debian mirrors occasionally serve mismatched files while they sync
RUN for i in 1 2 3 4 5; do \
      apt-get -o Acquire::Retries=5 update && \
      apt-get -o Acquire::Retries=5 install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 curl && break; \
      echo "apt attempt $i failed, retrying"; rm -rf /var/lib/apt/lists/*; sleep 20; \
    done && ffmpeg -version > /dev/null && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    TRUTHLENS_HOSTED=1 \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements-hosted.txt .
RUN pip install --no-cache-dir --user torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements-hosted.txt

COPY --chown=user . .
RUN mkdir -p weights cases \
    && [ -f weights/face_detection_yunet_2023mar.onnx ] || curl -L -o weights/face_detection_yunet_2023mar.onnx \
       https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
RUN python download_models.py

EXPOSE 7860
CMD ["python", "app.py"]
