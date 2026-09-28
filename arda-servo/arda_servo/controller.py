"""좌표 수신 → 각도 변환 → 서보 구동 메인 루프."""

import math
import time
from typing import Callable

from .angle import elevation_range, pan_angle_to_xyz, xyz_to_pan_angle
from .receiver import Coord, CoordReceiver
from .servo import PanServo
from .site import local_to_latlon
from .thermal_receiver import ThermalPan, ThermalPanReceiver
from .utils import get_logger, send_fall_report

logger = get_logger(__name__)


class ServoController:
    """평소엔 마지막으로 멈춘 각도에 그대로 있다가, 레이더가 낙하를
    확정(`fall=True`)한 좌표를 받을 때만 그 방향으로 움직여 `dwell_seconds`간
    머문다. `fall=False`인 일반 추적 좌표는 무시한다 — 낙하 시점에만 그
    지점을 확인하는 용도이지, 사람을 계속 따라다니는 용도가 아니기 때문이다.
    dwell이 어떻게 끝나든(시간 초과·열화상 포기·열화상 확정) 서보는 홈으로
    돌아가지 않고 그 자리에 멈춘다 — 오직 레이더 트리거만 서보를 움직인다.
    `run_forever()` 시작 시 1회 `center_deg`로 이동하는 것만 예외다.

    `thermal_receiver`가 주어지면 dwell 중에만 열화상(arda-thermal-test)의
    발열 방향 보정을 받아 그 방향으로 각도를 더 움직이고 dwell을 연장한다
    — 열원이 보이는 동안 계속 따라가다가 보정이 끊기면 dwell이 자연히
    만료돼 멈춘다. 보정에 `vertical_offset`(프레임 세로 편차)이 함께 오고
    카메라 설치 정보(`install_height_m`/`camera_tilt_deg`/`vertical_fov_deg`)가
    갖춰져 있으면 매 보정마다 거리·좌표를 다시 계산해 로그로 남긴다.

    열화상이 포기(`give_up`)하거나 확정(`confirmed`)하면 dwell 만료를
    기다리지 않고 즉시 끝내고(각도는 유지) 레이더 트리거를 다시 받는다.
    확정 시에는 이번 dwell을 시작시킨 레이더 원좌표(x, y)와 열화상 추적
    후 최종 좌표를 함께 로그로 남긴다 — 거리는 카메라 설치 정보와
    `vertical_offset`이 모두 있으면 "카메라가 z=0 평면(지면/수면)을 보고
    있다"는 가정으로 `elevation_range()`가 매번 새로 역산하고(사람이
    앞뒤로도 움직였을 가능성을 반영), 없으면 레이더 원좌표의 거리를
    그대로 쓰고 방향만 최종 각도로 바꾼다.

    `site_lat`/`site_lon`이 주어지면(arda-radar가 GPS 위경도로 낙하 위치를
    보고하는 방식과 맞추기 위함) 이 로그의 좌표를 로컬 미터 대신
    `local_to_latlon()`으로 변환한 위도/경도로 남긴다. 레이더 원좌표와
    열화상 추적 후 좌표 모두 이미 "레이더 로컬 좌표계" 기준이라 같은
    `site_lat`/`site_lon`/`site_heading_deg`로 변환하면 arda-radar의 GPS
    로그와 그대로 비교할 수 있다.

    `report_url`이 함께 주어지면(그리고 `site_lat`/`site_lon`으로 위경도
    변환이 가능하면), 열화상 확정(`pan.confirmed`) 시 이 열화상 추적 후
    보정 좌표를 `send_fall_report()`로 그 URL에 POST한다 — arda-radar가
    처음 감지한 대략적인 좌표가 아니라, 여기서 계산한 최종 보정 좌표가
    나간다(레이더 최초 감지 좌표를 아는 쪽은 arda-radar이지만, 그 좌표를
    열화상 추적으로 보정한 최종 값을 아는 쪽은 이 클래스뿐이기 때문이다).

    arda-radar가 낙하마다 신뢰도(`Coord.confidence`, 0~1)를 함께 보내는데,
    dwell 중 지금 쫓는 후보보다 confidence가 더 높은 새 낙하 좌표가 오면
    진행 중이던 추적을 버리고 그 즉시 새 좌표로 재조준해 dwell을 처음부터
    다시 시작한다 — 더 유력한 후보가 나타났으면 기존 위치를 붙잡고 있을
    이유가 없기 때문이다. 단, 열화상이 이번 dwell에서 원하는 모양과 매칭된
    보정(`ThermalPan.matched=True`, give_up/confirmed 아닌 일반 보정)을 한
    번이라도 보내온 뒤(`_thermal_engaged`)라면 confidence와 무관하게 이
    선점을 무시한다 — 단순히 열이 감지됐다는 것만으로는(matched=False)
    제어권이 넘어가지 않는다. 레이더 좌표는 대략적인 초기 조준일 뿐이고,
    열화상이 원하는 모양과 매칭되기 시작한 순간부터는 서보 제어권을
    열화상이 갖는다는 원칙이다. 열화상이 스스로 끝내야만(포기/확정) 다시
    레이더 좌표를 선점 대상으로 받아들인다.
    """

    def __init__(
        self,
        servo: PanServo,
        receiver: CoordReceiver | None = None,
        center_deg: float = 90.0,
        invert: bool = False,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        dwell_seconds: float = 3.0,
        thermal_receiver: ThermalPanReceiver | None = None,
        thermal_pan_gain_deg: float = 8.0,
        install_height_m: float | None = None,
        camera_tilt_deg: float | None = None,
        vertical_fov_deg: float | None = None,
        site_lat: float | None = None,
        site_lon: float | None = None,
        site_heading_deg: float = 0.0,
        report_url: str | None = None,
        on_confirmed: Callable[[float, float], None] | None = None,
    ):
        self._servo = servo
        self._receiver = receiver
        self._center_deg = center_deg
        self._invert = invert
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._dwell_seconds = dwell_seconds
        self._dwell_until = 0.0
        self._thermal_receiver = thermal_receiver
        self._thermal_pan_gain_deg = thermal_pan_gain_deg
        self._install_height_m = install_height_m
        self._camera_tilt_deg = camera_tilt_deg
        self._vertical_fov_deg = vertical_fov_deg
        self._site_lat = site_lat
        self._site_lon = site_lon
        self._site_heading_deg = site_heading_deg
        self._report_url = report_url
        self._on_confirmed = on_confirmed
        # 현재 dwell을 시작시킨 레이더 원좌표와 그 confidence — 전자는 열화상이
        # confirmed를 보낼 때 최종 각도(및 가능하면 거리)를 좌표로 역산해 이
        # 값과 비교 로그를 남기기 위함, 후자는 더 높은 확률의 새 낙하 후보가
        # 왔을 때 선점 여부를 판단하기 위함.
        self._dwell_start_coord: Coord | None = None
        self._dwell_confidence: float = 0.0
        # 이번 dwell에서 열화상이 원하는 모양과 매칭된 일반 보정(matched=True)을
        # 한 번이라도 보내왔는지 — True가 되면 레이더의 confidence 기반 선점을
        # 막는다(원하는 모양과 매칭되기 시작한 뒤로는 열화상이 서보 제어권을
        # 갖는다는 원칙). 단순히 열이 감지된 것(matched=False)만으로는 True가
        # 되지 않는다.
        self._thermal_engaged: bool = False

    def run_forever(self) -> None:
        """UDP로 좌표를 수신하며 서보를 구동 (정상 운영 모드)."""
        if self._receiver is None:
            raise RuntimeError("run_forever()에는 receiver가 필요합니다 — 수동 모드는 run_manual()을 사용하세요")

        self._servo.set_angle(self._center_deg)
        logger.info("서보 제어기 시작 — 홈 포지션(%.1f°)에서 낙하 트리거 대기 중", self._center_deg)
        try:
            while True:
                self.step()
        except KeyboardInterrupt:
            logger.info("사용자 중단")
        finally:
            self._servo.close()
            self._receiver.close()
            if self._thermal_receiver is not None:
                self._thermal_receiver.close()

    def run_manual(self) -> None:
        """표준입력으로 'x y' 좌표를 직접 입력받아 서보를 구동한다.

        레이더(arda-radar)나 UDP 연결 없이 서보 하드웨어/배선만 단독으로
        테스트하기 위한 모드다.
        """
        logger.info("수동 입력 모드 시작 — 'x y' 형식으로 좌표 입력 (예: 0.5 1.2), q 입력 시 종료")
        try:
            while True:
                try:
                    line = input("x y > ").strip()
                except EOFError:
                    break

                if line.lower() in ("q", "quit", "exit"):
                    break
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 2:
                    print("형식 오류 — 'x y' 두 값을 공백으로 구분해 입력하세요 (예: 0.5 1.2)")
                    continue

                try:
                    x, y = float(parts[0]), float(parts[1])
                except ValueError:
                    print("숫자로 입력해주세요 (예: 0.5 1.2)")
                    continue

                angle = xyz_to_pan_angle(
                    x, y,
                    center_deg=self._center_deg,
                    min_deg=self._servo.min_deg,
                    max_deg=self._servo.max_deg,
                    invert=self._invert,
                    offset_x=self._offset_x,
                    offset_y=self._offset_y,
                )
                self._servo.set_angle(angle)
                print(f"→ angle = {angle:.1f}°")
        except KeyboardInterrupt:
            logger.info("사용자 중단")
        finally:
            self._servo.close()

    def step(self) -> None:
        """수신 대기 1회 + (낙하 좌표면) 서보 이동. 테스트/단위 실행용으로 분리."""
        pan = self._thermal_receiver.recv() if self._thermal_receiver is not None else None
        coord = self._receiver.recv()
        now = time.time()

        if now < self._dwell_until:
            # 지금 쫓는 후보보다 confidence가 높은 새 낙하 좌표가 오면 즉시
            # 그쪽으로 전환한다(선점) — 단, 열화상이 이미 열원을 잡아 추적
            # 중이면(_thermal_engaged) 무시하고 열화상이 제어권을 유지한다.
            higher_confidence_coord = (
                coord is not None and coord.fall and coord.confidence > self._dwell_confidence
            )
            if higher_confidence_coord and not self._thermal_engaged:
                logger.warning(
                    "[제어권 이동] 더 높은 확률의 낙하 후보 수신(%.2f > %.2f) — 기존 추적 중단, 즉시 재조준",
                    coord.confidence, self._dwell_confidence,
                )
                self._start_tracking(coord, now)
                return
            elif higher_confidence_coord:
                logger.info(
                    "[제어권 유지] 더 높은 확률의 낙하 후보(%.2f > %.2f) 수신 — 열화상이 이미 "
                    "매칭된 대상을 추적 중이라 무시함",
                    coord.confidence, self._dwell_confidence,
                )

            if pan is not None and (pan.give_up or pan.confirmed):
                self._end_tracking(pan)
            elif pan is not None:
                # 보정(팬)은 열이 감지된 모든 프레임에서 오지만, 제어권
                # (_thermal_engaged)은 원하는 모양과 매칭된 프레임에서만 걸린다
                # — 단순히 열이 감지된 것만으로 더 유력한 새 낙하 후보의
                # 선점을 막지 않기 위함(클래스 docstring 참고).
                if pan.matched:
                    self._thermal_engaged = True
                self._apply_thermal_pan(pan.offset)
                self._dwell_until = now + self._dwell_seconds

                coord = self._dwell_start_coord
                if coord is not None:
                    range_m, range_note = self._resolve_range(coord, pan.vertical_offset)
                    position = self._describe_position(self._servo.angle, range_m)
                    logger.info(
                        "열화상 보정 반영 — offset=%.2f → angle=%.1f°, 추적 연장 | "
                        "현재 %s (거리 %.2fm, %s)",
                        pan.offset, self._servo.angle, position, range_m, range_note,
                    )
                else:
                    logger.info(
                        "열화상 보정 반영 — offset=%.2f → angle=%.1f°, 추적 연장",
                        pan.offset, self._servo.angle,
                    )
            elif coord is not None:
                logger.debug("dwell 중 — 좌표 무시 (남은 %.1fs)", self._dwell_until - now)
            return

        if self._dwell_until:
            self._dwell_until = 0.0
            logger.info(
                "dwell 시간 초과로 종료 — 각도 %.1f°에서 정지, 레이더 트리거 재개",
                self._servo.angle,
            )
            self._dwell_start_coord = None
            self._dwell_confidence = 0.0
            self._thermal_engaged = False

        if coord is None or not coord.fall:
            # 낙하가 아닌 일반 추적 좌표는 무시한다 — 평소엔 홈 포지션에
            # 고정해 지정된 구역(강/바다 등)을 보고 있어야 하기 때문이다.
            return

        self._start_tracking(coord, now)

    def _start_tracking(self, coord: Coord, now: float) -> None:
        """레이더 낙하 좌표로 서보를 이동시키고 dwell을 (재)시작함 — 홈에서
        새로 낙하를 받을 때와, dwell 중 더 높은 확률의 후보로 선점 전환할
        때 둘 다에서 쓰인다. 이전에 추적하던 좌표·열화상 진행 상황(있었다면)은
        전부 버리고 이 좌표 기준으로 처음부터 다시 시작한다."""
        angle = xyz_to_pan_angle(
            coord.x,
            coord.y,
            center_deg=self._center_deg,
            min_deg=self._servo.min_deg,
            max_deg=self._servo.max_deg,
            invert=self._invert,
            offset_x=self._offset_x,
            offset_y=self._offset_y,
        )
        self._servo.set_angle(angle)
        self._dwell_start_coord = coord
        self._dwell_confidence = coord.confidence
        self._thermal_engaged = False
        logger.info(
            "낙하 좌표 수신 — x=%.2f y=%.2f confidence=%.2f → angle=%.1f°로 이동, "
            "%.1fs간 정지(dwell), 열화상 판정 대기",
            coord.x, coord.y, coord.confidence, angle, self._dwell_seconds,
        )
        if self._dwell_seconds > 0:
            self._dwell_until = now + self._dwell_seconds

    def _apply_thermal_pan(self, offset: float) -> None:
        """열화상이 보낸 정규화 편차(-1.0~1.0)만큼 현재 각도에서 더 회전함."""
        if self._invert:
            offset = -offset
        current = self._servo.angle if self._servo.angle is not None else self._center_deg
        new_angle = current + offset * self._thermal_pan_gain_deg
        self._servo.set_angle(new_angle)

    def _end_tracking(self, pan: ThermalPan) -> None:
        """열화상의 give_up/confirmed 신호를 받아 dwell을 즉시 끝냄. 서보는
        홈으로 복귀하지 않고 지금 각도에 그대로 멈춘다 — 트리거(레이더
        낙하 좌표)가 있을 때만 움직이고, 그 외에는 항상 마지막 위치를
        유지한다."""
        final_angle = self._servo.angle
        self._dwell_until = 0.0

        if pan.confirmed:
            coord = self._dwell_start_coord
            if coord is not None:
                range_m, range_note = self._resolve_range(coord, pan.vertical_offset)
                orig_position = self._describe_xy(coord.x, coord.y)
                final_x, final_y = pan_angle_to_xyz(
                    final_angle, range_m,
                    center_deg=self._center_deg, invert=self._invert,
                    offset_x=self._offset_x, offset_y=self._offset_y,
                )
                final_position = self._describe_xy(final_x, final_y)
                position_word = "위치" if self._site_lat is not None else "좌표"
                logger.warning(
                    "열화상 사람 확정 — 레이더 원%s %s → 열화상 추적 후 %s %s "
                    "(거리 %.2fm, %s) — 각도 %.1f°에서 정지, 레이더 트리거 재개",
                    position_word, orig_position, position_word, final_position,
                    range_m, range_note, final_angle,
                )

                # 레이더 최초 감지 좌표가 아니라, 여기서 방금 계산한 열화상 추적
                # 후 보정 좌표(final_x, final_y)를 알린다 — arda-radar는 더
                # 이상 이 보고를 하지 않는다(main.py 참고). report_url은 이
                # 좌표를 HTTP POST로 보내고 싶은 호출자(arda-raset 등)를
                # 위한 것이고, on_confirmed는 HTTP가 아닌 다른 경로(예:
                # arda-bringup의 ROS 토픽 발행)로 알리고 싶은 호출자를 위한
                # 콜백이다 — 둘 다, 어느 한쪽만, 또는 아무것도 설정하지 않을
                # 수 있다.
                if self._report_url or self._on_confirmed:
                    final_latlon = self._resolve_latlon(final_x, final_y)
                    if final_latlon is not None:
                        final_lat, final_lon = final_latlon
                        if self._report_url:
                            sent = send_fall_report(self._report_url, final_lat, final_lon)
                            logger.info(
                                "%s — 보정 좌표 lat=%.6f lon=%.6f -> %s",
                                "낙하 위치 보고 전송됨" if sent else "낙하 위치 보고 전송 실패",
                                final_lat, final_lon, self._report_url,
                            )
                        if self._on_confirmed:
                            try:
                                self._on_confirmed(final_lat, final_lon)
                            except Exception:
                                logger.exception("on_confirmed 콜백 실행 중 오류")
                    else:
                        logger.warning(
                            "report_url/on_confirmed이 설정됐지만 site.lat/site.lon이 "
                            "없어 위경도로 변환할 수 없음 — 보고 전송 생략"
                        )
            else:
                logger.warning(
                    "열화상 사람 확정 — 레이더 원좌표 없음, 각도 %.1f°에서 정지, 레이더 트리거 재개",
                    final_angle,
                )
        else:
            logger.info(
                "열화상이 추적을 포기함 — 각도 %.1f°에서 정지, 레이더 트리거 재개",
                final_angle,
            )

        self._dwell_start_coord = None
        self._dwell_confidence = 0.0
        self._thermal_engaged = False

    def _resolve_range(self, coord: Coord, vertical_offset: float | None) -> tuple[float, str]:
        """현재 거리를 구함. 가능하면 카메라 설치 정보로 z=0 기준 역산하고,
        정보가 부족하면 레이더 원좌표의 거리를 그대로 쓴다. (거리, 설명) 반환.
        확정 시점뿐 아니라 dwell 추적 중 매 보정마다도 호출된다."""
        geometry_ready = (
            self._install_height_m is not None
            and self._camera_tilt_deg is not None
            and self._vertical_fov_deg is not None
        )
        radar_range_m = math.hypot(coord.x - self._offset_x, coord.y - self._offset_y)

        if not geometry_ready or vertical_offset is None:
            return radar_range_m, "레이더 원거리 유지, 방향만 갱신"

        try:
            range_m = elevation_range(
                vertical_offset, self._install_height_m, self._camera_tilt_deg, self._vertical_fov_deg,
            )
            return range_m, "z=0 평면 기준 열화상 세로 위치로 거리 역산"
        except ValueError as e:
            logger.warning("거리 역산 실패(%s) — 레이더 원거리로 대체", e)
            return radar_range_m, "레이더 원거리로 대체"

    def _resolve_latlon(self, x: float, y: float) -> tuple[float, float] | None:
        """로컬 (x, y)를 site_lat/site_lon이 설정돼 있으면 위경도로 변환해
        반환하고, 없으면 None(로컬 좌표만 있고 GPS 변환은 불가능하다는 뜻).
        report_url 전송에 실제로 쓸 숫자 좌표가 필요할 때 이걸 쓴다 —
        `_describe_xy()`는 로그용 문자열만 주기 때문에 따로 뺐다."""
        if self._site_lat is None or self._site_lon is None:
            return None
        return local_to_latlon(x, y, self._site_lat, self._site_lon, self._site_heading_deg)

    def _describe_xy(self, x: float, y: float) -> str:
        """로컬 (x, y)를 site 설정이 있으면 위경도, 없으면 로컬 좌표 문자열로 변환함."""
        latlon = self._resolve_latlon(x, y)
        if latlon is not None:
            lat, lon = latlon
            return f"lat={lat:.6f} lon={lon:.6f}"
        return f"x={x:.2f} y={y:.2f}"

    def _describe_position(self, angle: float, range_m: float) -> str:
        """서보 각도·거리로 좌표를 구해 _describe_xy()로 문자열화함."""
        x, y = pan_angle_to_xyz(
            angle, range_m,
            center_deg=self._center_deg, invert=self._invert,
            offset_x=self._offset_x, offset_y=self._offset_y,
        )
        return self._describe_xy(x, y)
