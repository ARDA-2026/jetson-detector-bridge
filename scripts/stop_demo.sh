#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if systemctl cat arda-jetson.service >/dev/null 2>&1; then
    sudo systemctl stop arda-jetson.service
fi
"$ROOT/scripts/demo_stop.sh"
# DHCP 주소는 다음 시연에 바뀔 수 있으므로 임시 URL을 지운다.
if [[ -f /run/arda-jetson/report-url ]]; then
    sudo rm -f /run/arda-jetson/report-url
fi
