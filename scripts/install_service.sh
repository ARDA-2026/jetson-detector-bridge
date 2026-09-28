#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if (( EUID == 0 )); then echo "일반 사용자로 실행하세요: ./scripts/install_service.sh" >&2; exit 1; fi
if [[ "$ROOT" == *' '* || "$ROOT" == *'@'* ]]; then echo "설치 경로에 공백이나 @는 사용할 수 없습니다" >&2; exit 1; fi
[[ -x "$ROOT/arda-raset/.venv/bin/python" ]] || { echo "먼저 uv sync --frozen을 실행하세요" >&2; exit 1; }
USER_NAME="$(id -un)"
GROUP_NAME="$(id -gn)"
python3 - "$ROOT" "$USER_NAME" "$GROUP_NAME" <<'PY'
from pathlib import Path
import sys
root, user, group = sys.argv[1:]
template = (Path(root) / 'services/arda-jetson.service.in').read_text()
unit = template.replace('@ROOT@', root).replace('@USER@', user).replace('@GROUP@', group)
(Path(root) / 'services/arda-jetson.service').write_text(unit)
PY
sudo systemctl link "$ROOT/services/arda-jetson.service"
sudo systemctl daemon-reload
sudo systemctl enable arda-jetson.service
echo "서비스를 등록했습니다. 지금 실행하려면 sudo systemctl start arda-jetson.service"
