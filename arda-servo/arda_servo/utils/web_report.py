"""열화상 추적으로 보정된 최종 낙하 위치를 웹 엔드포인트로 전송.

arda-radar(arda/utils/web_report.py)의 send_fall_report()와 동일한 함수다.
arda-servo는 arda-radar와 완전히 분리된 프로젝트라 코드를 직접 참조하지
않고 이 파일에 그대로 복사해서 쓴다(site.py와 같은 원칙) — config/
settings.yaml의 site.report_url 값도 arda-radar 쪽 site 설정과 같은
값으로 수동으로 맞춰야 한다.

호출 주체가 arda-radar에서 arda-servo로 옮겨온 이유: report_url로 보내는
"열화상이 사람으로 확인한 최종 위치"는 열화상 추적으로 보정된 좌표여야
하는데, 그 보정 좌표(서보 최종 각도 + 역산 거리)를 아는 곳은
ServoController(_end_tracking())뿐이다. arda-radar는 낙하를 처음 감지한
대략적인 좌표만 알기 때문에, 예전처럼 arda-radar가 이 함수를 호출하면
보정되지 않은 좌표가 나간다.

{"lat": <위도>, "lon": <경도>, "timestamp": "<한국시간 ISO8601>"}

image_jpeg는 arda-servo에는 카메라 이미지가 없어 항상 생략하지만, 시그니처는
arda-radar 쪽과 그대로 맞춰뒀다 — 두 파일을 나란히 비교하기 쉽게 하기 위함.
"""

import base64
import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from .logger import get_logger

logger = get_logger(__name__)

KST = timezone(timedelta(hours=9))


def send_fall_report(
    url: str,
    lat: float,
    lon: float,
    image_jpeg: bytes | None = None,
    confirmed: bool = False,
    timeout: float = 3.0,
) -> bool:
    """{"lat", "lon", "timestamp"}(한국시간) JSON을 url로 POST한다.

    네트워크 문제로 서보 제어 루프가 죽으면 안 되므로, 실패해도 예외를
    던지지 않고 False만 반환한다.
    """
    payload_dict = {
        "lat": lat,
        "lon": lon,
        "timestamp": datetime.now(KST).isoformat(),
    }
    if image_jpeg is not None:
        payload_dict["thermal_image_base64"] = base64.b64encode(image_jpeg).decode("ascii")
        payload_dict["confirmed"] = confirmed
    payload = json.dumps(payload_dict).encode("utf-8")

    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            ok = 200 <= response.status < 300
            if not ok:
                logger.warning("낙하 위치 전송 실패 — HTTP %d", response.status)
            return ok
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        logger.warning("낙하 위치 전송 실패 — %s", e)
        return False
