FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget unzip \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    libc-dev \
    libmupdf-dev \
    libglib2.0-dev \
    pkg-config \
    python3-dev \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz \
    && tar -xvzf ngrok-v3-stable-linux-arm64.tgz -C /usr/local/bin \
    && rm ngrok-v3-stable-linux-arm64.tgz

RUN ngrok version
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade openai httpx gradio selenium webdriver-manager


COPY . .


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]