#!/usr/bin/env bash
set -euo pipefail

REPO_URL="git@github.com:moux1024/S-OCR.git"
DEPLOY_DIR="/opt/ocr-service"

echo "==> Cloning/updating repository"
if [ -d "${DEPLOY_DIR}/.git" ]; then
    cd "${DEPLOY_DIR}"
    git pull
else
    git clone "${REPO_URL}" "${DEPLOY_DIR}"
    cd "${DEPLOY_DIR}"
fi

echo "==> Setting up venv and installing dependencies"
if [ ! -d venv ]; then
    python3 -m venv venv
fi
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q -r requirements.txt

echo "==> Installing systemd service"
cp "${DEPLOY_DIR}/ocr-service.service" /etc/systemd/system/ocr-service.service
systemctl daemon-reload
systemctl enable ocr-service
systemctl restart ocr-service
sleep 2
systemctl status ocr-service --no-pager

echo "==> Checking health"
curl -sf http://localhost:8000/health | python3 -m json.tool || echo "WARNING: health check failed"

echo "==> Deploy complete"
