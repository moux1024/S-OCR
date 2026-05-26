#!/usr/bin/env bash
set -euo pipefail

SERVER="root@101.201.33.125"
REMOTE_DIR="/opt/ocr-service"

echo "==> Syncing files to ${SERVER}:${REMOTE_DIR}"
ssh "${SERVER}" "mkdir -p ${REMOTE_DIR}"
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
    "$(dirname "$0")/" "${SERVER}:${REMOTE_DIR}/"

echo "==> Setting up venv and installing dependencies"
ssh "${SERVER}" bash -s <<REMOTE
set -euo pipefail
cd ${REMOTE_DIR}
if [ ! -d venv ]; then
    python3 -m venv venv
fi
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q -r requirements.txt
REMOTE

echo "==> Installing systemd service"
ssh "${SERVER}" bash -s <<REMOTE
set -euo pipefail
cp ${REMOTE_DIR}/ocr-service.service /etc/systemd/system/ocr-service.service
systemctl daemon-reload
systemctl enable ocr-service
systemctl restart ocr-service
sleep 2
systemctl status ocr-service --no-pager
REMOTE

echo "==> Checking health"
curl -sf http://101.201.33.125:8000/health | python3 -m json.tool || echo "WARNING: health check failed"

echo "==> Deploy complete"
