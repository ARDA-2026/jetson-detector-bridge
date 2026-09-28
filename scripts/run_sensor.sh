#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/config/runtime.env"
# 시연 중 CLI로 지정한 Windows 주소는 재부팅 전까지만 우선 적용한다.
if [[ -f /run/arda-jetson/report-url ]]; then
  IFS= read -r ARDA_REPORT_URL < /run/arda-jetson/report-url || true
fi
PYTHON="$ROOT/arda-raset/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "Python 환경이 없습니다. README의 uv sync 단계를 완료하세요." >&2; exit 1; }
args=(
  --report-url "${ARDA_REPORT_URL:-}"
  --site-lat "${ARDA_SITE_LAT:-37.5336}"
  --site-lon "${ARDA_SITE_LON:-126.9364}"
  --site-heading-deg "${ARDA_SITE_HEADING_DEG:-320.0}"
  --radar-cli-port "${ARDA_RADAR_CLI_PORT:-/dev/ttyUSB0}"
  --radar-data-port "${ARDA_RADAR_DATA_PORT:-/dev/ttyUSB1}"
  --radar-settings "$ROOT/arda-radar/config/settings.yaml"
  --radar-profile "$ROOT/arda-radar/config/profiles/xwr68xx_AOP_profile_short_range.cfg"
  --servo-config "$ROOT/arda-servo/config/settings.yaml"
)
[[ "${ARDA_YOLO:-0}" == 1 ]] && args+=(--yolo)
[[ "${ARDA_SIMULATE_SERVO:-0}" == 1 ]] && args+=(--simulate-servo)
[[ "${ARDA_SIMULATE_THERMAL:-0}" == 1 ]] && args+=(--simulate-thermal)
[[ "${ARDA_NO_RADAR:-0}" == 1 ]] && args+=(--no-radar)
cd "$ROOT/arda-raset"
exec "$PYTHON" main.py "${args[@]}" "$@"
