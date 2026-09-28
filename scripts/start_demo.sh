#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# 무선 SSH가 AP 전환 중 끊기므로 센서 서비스를 먼저 준비한다.
if systemctl cat arda-jetson.service >/dev/null 2>&1; then
    sudo systemctl start arda-jetson.service
    sudo systemctl is-active --quiet arda-jetson.service || { echo "센서 서비스가 실행되지 않았습니다. journalctl -u arda-jetson.service를 확인하세요." >&2; exit 1; }
else
    echo "센서 서비스 미등록: scripts/install_service.sh를 실행하거나 scripts/run_sensor.sh를 별도 터미널에서 실행하세요." >&2
fi
"$ROOT/scripts/demo_start.sh"
