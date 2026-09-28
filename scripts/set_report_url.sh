#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# != 1 || ! "$1" =~ ^https?://[^[:space:]]+/report$ ]]; then
    echo "사용법: $0 http://<Windows-IP>:8000/report" >&2
    exit 2
fi
if ! systemctl cat arda-jetson.service >/dev/null 2>&1; then
    echo "arda-jetson.service가 설치되지 않았습니다. 수동 실행은 scripts/run_sensor.sh --report-url URL을 사용하세요." >&2
    exit 1
fi
sudo install -d -m 0755 /run/arda-jetson
printf '%s\n' "$1" | sudo tee /run/arda-jetson/report-url >/dev/null
sudo systemctl restart arda-jetson.service
systemctl is-active arda-jetson.service
printf '웹 전송 주소: %s\n' "$1"
