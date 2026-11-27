FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    ORT_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    HF_HOME=/app/.cache/huggingface \
    TTS_PARADEE_MODEL_DIR=/app/models/paradee_thai

WORKDIR /app

# System runtime dependencies:
# - ffmpeg, libopus0: Discord voice playback/transcoding & Opus framing
# - libsndfile1: Required by soundfile (PyThaiTTS) on slim x86_64/ARM64 Linux
# - fonts-tlwg-garuda-otf, fonts-noto-core, fonts-dejavu-core: Thai/English manga typesetting
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libopus0 \
    libsndfile1 \
    fonts-tlwg-garuda-otf \
    fonts-noto-core \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the 8.07 MB MakHamTTS Paradee 1.1 INT8 ONNX model + config into /app/models/paradee_thai
# Placed BEFORE `COPY . .` so Docker caches this layer across normal code deploys.
# Best-effort (`|| true`) so a temporary Hugging Face Hub hiccup never fails a Render deploy.
RUN mkdir -p /app/models/paradee_thai && python3 -c "\
import os, shutil; \
from huggingface_hub import hf_hub_download; \
out = '/app/models/paradee_thai'; \
m = hf_hub_download('pythainlp/MakHamTTS', 'paradee_thai/paradee_thai_int8.onnx'); \
c = hf_hub_download('pythainlp/MakHamTTS', 'paradee_thai/config.json'); \
shutil.copy2(m, os.path.join(out, 'paradee_thai_int8.onnx')); \
shutil.copy2(c, os.path.join(out, 'config.json')); \
print('Baked MakHamTTS Paradee INT8 model successfully')" || true

COPY . .

EXPOSE 10000

CMD ["python", "bot.py"]