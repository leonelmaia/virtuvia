#!/bin/sh

if [ -z "$NGROK_AUTHTOKEN" ]; then
  echo "⚠️  NGROK_AUTHTOKEN not set! Ngrok won't be authenticated."
else
  echo "Configuring ngrok with authtoken..."
  /usr/local/bin/ngrok authtoken $NGROK_AUTHTOKEN
fi

# Start ngrok tunnel for port 7860 (Gradio)
ngrok http 7860 --log=stdout --log-format=json &

# Run vitruvia.py with unbuffered stdout/stderr
python -u vitruvia.py

# Start FastAPI on port 8000 in background
uvicorn law_api:app --host 0.0.0.0 --port 8000 &

# Start your Gradio chatbot (blocking call)
python chat_interface.py
