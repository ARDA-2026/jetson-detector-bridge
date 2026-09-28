"""레이더 프로세스(arda-radar)로부터 UDP로 좌표를 수신."""

import json
import socket
from dataclasses import dataclass

from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class Coord:
    x: float
    y: float
    z: float
    fall: bool
    ts: float
    confidence: float = 0.0  # 0~1, FallDetector.last_fall_confidence 그대로. dwell 중 선점 판단에 쓰임


class CoordReceiver:
    """레이더 좌표 UDP 수신기.

    소켓에 타임아웃을 두어 recv()가 주기적으로 리턴하도록 한다 — 좌표가
    끊겨도 메인 루프가 블로킹되지 않고 대기 상태를 감지할 수 있다.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9999, timeout: float = 0.5):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(timeout)

    def recv(self) -> Coord | None:
        """좌표 1건 수신. timeout 내 수신 실패 또는 잘못된 패킷이면 None."""
        try:
            data, _ = self._sock.recvfrom(4096)
        except socket.timeout:
            return None

        try:
            obj = json.loads(data.decode("utf-8"))
            return Coord(
                x=float(obj["x"]),
                y=float(obj["y"]),
                z=float(obj["z"]),
                fall=bool(obj.get("fall", False)),
                ts=float(obj.get("ts", 0.0)),
                confidence=float(obj.get("confidence", 0.0)),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("잘못된 좌표 패킷 수신: %s", e)
            return None

    def close(self) -> None:
        self._sock.close()
