#!/usr/bin/env bash
# Run RoomSketch locally.
#   ./run.sh           http://localhost:8000 (desktop browser, uploads, demo)
#   ./run.sh --https   https://<your-LAN-IP>:8443 with a self-signed cert, so an
#                      iPhone on the same Wi-Fi can use the camera capture.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt -r requirements-photogrammetry.txt
fi
if [ "${1:-}" = "--https" ]; then
  mkdir -p certs
  if [ ! -f certs/cert.pem ]; then
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=roomsketch.local" \
      -keyout certs/key.pem -out certs/cert.pem >/dev/null 2>&1
  fi
  IP=$( (ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}') || true)
  echo "Open https://${IP:-<this-computer-ip>}:8443 on your phone (accept the certificate warning)."
  exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8443 \
    --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
fi
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
