FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install audio codecs required for Discord voice channels, edge-tts, STT conversion, and fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libopus0 \
    libopus-dev \
    libsodium-dev \
    flac \
    fonts-noto-core \
    fonts-tlwg-garuda-ttf \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY bot.py .
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:10000/healthz', timeout=3)" || exit 1

# Render injects the PORT environment variable dynamically (defaults to 10000)
EXPOSE 10000

CMD ["python", "bot.py"]