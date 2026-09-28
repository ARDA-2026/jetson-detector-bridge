"""열화상 판정기(arda-thermal-test)로부터 발열 방향 보정값을 UDP로 수신.

레이더 좌표(receiver.py)로 이미 이동한 이후, dwell 중 열원이 계속 감지되는
동안 그 방향으로 서보를 미세 추적하기 위한 채널이다. 좌표(x,y,z)가 아니라
열화상 프레임 중심 대비 좌우 편차(offset)와 세로 편차(vertical_offset)를
정규화된 -1.0~1.0 값으로 받는다 — 실제 각도로의 변환(gain, invert)과
거리·좌표 역산은 controller.py가 담당한다. vertical_offset은 매 보정마다
함께 올 수 있어, dwell 추적 중에도 카메라 설치 정보로 거리·좌표를 계속
갱신할 수 있다.

일반 보정에는 "matched"(그 프레임이 원하는 모양과 매칭됐는지) 플래그가
함께 온다 — 열이 감지된 모든 프레임에서 보정 자체는 오지만(카메라가 계속
열원 방향을 향하게), 서보가 "제어권을 넘겨받았다"고 볼지(`_thermal_engaged`)
는 매칭된 프레임인지로만 판단한다. 단순히 배경보다 뜨거운 것이 있다는
것만으로는(matched=False) 제어권을 넘기지 않는다 — 사람 모양이 아닌 열원
(반사광, 손 등)에 의해 더 유력한 새 낙하 후보의 선점이 막히지 않도록
하기 위함이다. 구버전 송신측(matched 필드 없음)은 기본값 False로 처리돼
기존처럼 전혀 engaged가 되지 않는다(안전한 방향의 하위호환).

열화상이 관찰을 끝냈는데(사람 매칭이 dwell_seconds 동안 계속 실패해)
더 이상 보정을 보낼 수 없는 경우에는 {"give_up": true}를 보낸다 —
서보가 자체 dwell 타이머 만료를 수동적으로 기다리지 않고 그 즉시 홈으로
복귀해 레이더가 바로 다음 낙하를 다시 제어할 수 있게 하기 위함이다.

반대로 사람 매칭에 성공해 관찰이 확정되면 {"confirmed": true, "vertical_offset": v}를
보낸다 — give_up과 마찬가지로 즉시 홈으로 복귀하되, vertical_offset(프레임
세로 중심 대비 편차, -1.0~1.0)이 있으면 카메라 설치 높이·기울기로 거리를
역산해 이번 낙하의 레이더 원좌표와 나란히 로그로 남긴다.
"""

import json
import socket
from dataclasses import dataclass

from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class ThermalPan:
    offset: float  # -1.0(왼쪽 끝) ~ 1.0(오른쪽 끝), 0.0이 프레임 중심
    ts: float
    give_up: bool = False  # True면 offset은 의미 없음 — 열화상이 추적을 포기했다는 신호
    confirmed: bool = False  # True면 offset은 의미 없음 — 열화상이 사람으로 확정했다는 신호
    vertical_offset: float | None = None  # -1.0(화면 위) ~ 1.0(화면 아래). 일반 보정/confirmed 모두에서 올 수 있음
    matched: bool = False  # 이 프레임이 원하는 모양과 매칭됐는지 — True일 때만 _thermal_engaged(제어권)가 걸림


class ThermalPanReceiver:
    """열화상 발열 방향 보정 UDP 수신기.

    메인 루프(레이더 좌표 수신)를 오래 막지 않도록 아주 짧은 타임아웃으로
    논블로킹 폴링한다.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9996, timeout: float = 0.01):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(timeout)

    def recv(self) -> ThermalPan | None:
        """보정값 1건 수신. 타임아웃 내 수신 실패 또는 잘못된 패킷이면 None."""
        try:
            data, _ = self._sock.recvfrom(4096)
        except socket.timeout:
            return None

        try:
            obj = json.loads(data.decode("utf-8"))
            if obj.get("give_up"):
                return ThermalPan(offset=0.0, ts=float(obj.get("ts", 0.0)), give_up=True)
            if obj.get("confirmed"):
                vertical_offset = obj.get("vertical_offset")
                if vertical_offset is not None:
                    vertical_offset = max(-1.0, min(1.0, float(vertical_offset)))
                return ThermalPan(
                    offset=0.0, ts=float(obj.get("ts", 0.0)), confirmed=True,
                    vertical_offset=vertical_offset,
                )
            offset = max(-1.0, min(1.0, float(obj["offset"])))
            vertical_offset = obj.get("vertical_offset")
            if vertical_offset is not None:
                vertical_offset = max(-1.0, min(1.0, float(vertical_offset)))
            return ThermalPan(
                offset=offset, ts=float(obj.get("ts", 0.0)), vertical_offset=vertical_offset,
                matched=bool(obj.get("matched", False)),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("잘못된 열화상 보정 패킷 수신: %s", e)
            return None

    def close(self) -> None:
        self._sock.close()
