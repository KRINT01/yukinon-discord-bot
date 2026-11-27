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

COPY *.py ./

# Run as non-root. Writes go only to /app (user_profiles.json) and the system temp dir.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Render injects the PORT environment variable dynamically (defaults to 10000)
EXPOSE 10000

CMD ["python", "bot.py"]