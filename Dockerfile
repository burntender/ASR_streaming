ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}
ARG INSTALL_TORCH=1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
 && if [ "$INSTALL_TORCH" = "1" ]; then python -m pip install torch --index-url "$TORCH_INDEX_URL"; fi \
 && python -m pip install -r /app/requirements.txt

COPY asr_core.py /app/asr_core.py
COPY api.py /app/api.py
COPY transcribe.py /app/transcribe.py
COPY entrypoint.sh /app/entrypoint.sh
COPY README.md /app/README.md

RUN chmod +x /app/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
