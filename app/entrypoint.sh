#!/bin/sh

if [ -z "$NGROK_AUTHTOKEN" ]; then
  echo "⚠️  NGROK_AUTHTOKEN não definido! O ngrok não será autenticado."
else
  echo "Configurando ngrok com authtoken..."
  /usr/local/bin/ngrok authtoken $NGROK_AUTHTOKEN
fi

# Starts tunelling through port 7860
ngrok http 7860 --log=stdout --log-format=json &

python chat_interface.py