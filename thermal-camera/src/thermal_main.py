"""레이더-서보 연동용 person 확정 메인 흐름 (온도 임계값 기반 판정).

arda-radar가 낙하를 감지하면 UDP로 트리거를 보내고, 이 스크립트는 그 트리거를
받아 최대 dwell_seconds 동안 열화상을 관찰해 사람 모양 발열 영역이 있는지
판정한 뒤, 결과를 다시 arda-radar로 UDP 회신한다. 카메라가 서보(arda-servo)에
고정 장착되어 서보가 향한 곳을 그대로 보므로, 트리거에는 좌표 없이 "지금
관찰을 시작하라"는 신호만 온다. 관찰 중 발열 위치를 매 프레임 arda-servo로
보내 서보가 그 방향을 계속 따라가게 하고, arda-radar가 더 높은 확률의 새
후보로 서보를 이미 옮겼다면 트리거 소켓에서 그 사실을 감지해 지금 관찰을
버리고 즉시 재시작한다.

발열 검출/구 형태 판정 로직(detect_hot_region/is_head_shape)은
thermal_head_detection.py에서 그대로 옮겨왔다 — 원본 파일은 수정하지 않았다.
같은 흐름을 YOLO 커스텀 모델로 판정하는 버전은 thermal_main_yolo.py 참고.

라즈베리파이/Jetson 모두 지원한다. 센서 코드는 Blinka가 플랫폼을 자동
인식해 그대로 동작하며, 가상환경만 라즈베리파이는 requirements.txt, Jetson은
requirements-jetson.txt로 나눠 쓴다 (README 23장 참고).
"""

import argparse
import json
import os
import socket
import sys
import time

import cv2
import numpy as np

# ============================================================
# 센서 설정값 (thermal_head_detection.py와 동일)
# ============================================================
# MLX90640 원본 해상도 — 칩 자체의 고정된 픽셀 배열이라 하우징이 어느
# 방향으로 장착되든 항상 이 모양(24행 x 32열)으로 나온다. 아래
# initialize_sensor()의 버퍼 크기/reshape에만 쓰고, 그 밖의 모든 곳
# (판정, 서보 보정, 컬러맵 등)에서는 실제 설치 방향을 반영한 WIDTH/HEIGHT를
# 쓴다.
RAW_WIDTH = 32
RAW_HEIGHT = 24

# 실치 재설치 후: 카메라 하우징이 원래 방향에서 시계방향으로 90도 돌아간
# 채로 고정됨 — read_frame()에서 그만큼을 반시계로 되돌려 다시 정방향으로
# 맞춘다. 카메라를 원래 방향으로 재장착하면 False로. 방향이 반대로
# 나오면(위아래/좌우가 뒤집혀 보이면) np.rot90 호출의 k=1을 k=-1로 바꿀 것
# (read_frame() 참고).
ROTATE_90_CCW = True

# 90/270도 회전은 가로세로가 서로 바뀐다 — WIDTH/HEIGHT/DISPLAY_WIDTH/
# DISPLAY_HEIGHT는 전부 "회전 보정을 마친 뒤" 실제 화면 기준이다. arda-servo
# config/settings.yaml의 camera_geometry.vertical_fov_deg도 이 회전 때문에
# 35도(원래 세로축)에서 55도(원래 가로축)로 같이 바꿔뒀다.
if ROTATE_90_CCW:
    WIDTH, HEIGHT = RAW_HEIGHT, RAW_WIDTH
    DISPLAY_WIDTH, DISPLAY_HEIGHT = 480, 640
else:
    WIDTH, HEIGHT = RAW_WIDTH, RAW_HEIGHT
    DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480

# adafruit_mlx90640 프레임은 렌즈 반대쪽(핀 쪽) 기준이라 실제로 보는 방향
# 기준으로는 좌우가 뒤집혀 나온다 — 그대로 쓰면 서보가 반대 방향으로 움직인다.
# 센서를 반대로 재장착하면 False로. 이 보정은 센서 칩 자체의 배선 특성이라
# ROTATE_90_CCW(하우징 장착 방향)와 무관하게 원본 좌표계에서 먼저 적용한다.
FLIP_HORIZONTAL = True

# 절대 온도 컬러맵 범위 (원본과 동일)
TEMP_MIN = 24.0
TEMP_MAX = 36.0

# 발열 영역 검출 설정값 (원본과 동일)
TEMP_OFFSET = 2.0
CORE_TEMP_OFFSET = 3.0
MIN_HOT_PIXELS = 4
MORPH_KERNEL_SIZE = 2

# 따뜻한 구 형태 판정 설정값 (원본과 동일, MLX90640 32x24 좌표 기준)
MIN_CIRCULARITY = 0.75
MIN_ASPECT_RATIO = 0.80
MAX_ASPECT_RATIO = 1.20
MIN_FILL_RATIO = 0.65
MAX_CORE_CENTER_OFFSET_RATIO = 0.60

SENSOR_WARMUP_FRAMES = 5
FRAME_INTERVAL_S = 0.5  # MLX90640 REFRESH_2_HZ 기준
WINDOW_NAME = "ARDA Thermal Main — 관찰 중"


# ============================================================
# 센서
# ============================================================
def initialize_sensor(simulate: bool = False):
    """I2C 버스와 MLX90640 센서를 초기화함. simulate=True면 하드웨어 없이
    배경(24도)+잡음만 있는 임의 프레임을 반환하는 폴백 함수를 대신 돌려준다
    (UDP/판정 파이프라인 검증용이며 실제 사람 형태는 아님)."""
    if simulate:
        print("[sensor] 시뮬레이션 모드로 실행")

        def read_frame_simulated():
            return 24.0 + np.random.normal(0, 0.3, size=(RAW_HEIGHT, RAW_WIDTH)).astype(np.float32)

        return None, read_frame_simulated

    try:
        import adafruit_mlx90640
        import board
        import busio
    except ImportError as exc:
        raise RuntimeError(
            "MLX90640 패키지를 불러오지 못했습니다. 가상환경에서 라즈베리파이는 "
            "'pip install -r requirements.txt'를, Jetson은 "
            "'pip install -r requirements-jetson.txt'를 실행하세요."
        ) from exc

    try:
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        sensor = adafruit_mlx90640.MLX90640(i2c)
        sensor.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
    except Exception as exc:
        raise RuntimeError(
            "MLX90640 초기화에 실패했습니다. I2C 활성화, 배선, "
            f"센서 주소 0x33을 확인하세요. 원인: {exc}"
        ) from exc

    buf = [0.0] * (RAW_WIDTH * RAW_HEIGHT)

    def read_frame_hardware():
        try:
            sensor.getFrame(buf)
        except ValueError:
            return None
        return np.asarray(buf, dtype=np.float32).reshape(RAW_HEIGHT, RAW_WIDTH)

    return i2c, read_frame_hardware


def read_frame(read_frame_fn) -> np.ndarray | None:
    """read_frame_fn()으로 얻은 원본(RAW_HEIGHT x RAW_WIDTH) 프레임에
    FLIP_HORIZONTAL(센서 칩 배선 특성) 보정을 먼저 적용한 뒤, ROTATE_90_CCW
    (실제 하우징 장착 방향) 보정을 적용해 WIDTH x HEIGHT 모양으로 돌려준다."""
    thermal = read_frame_fn()
    if thermal is None:
        return None
    if FLIP_HORIZONTAL:
        thermal = np.fliplr(thermal)
    if ROTATE_90_CCW:
        thermal = np.rot90(thermal, k=1)
    return thermal


# ============================================================
# 발열 영역 검출/판정 (thermal_head_detection.py의 detect_hot_region/
# is_head_shape를 그대로 이식 — 원본은 수정하지 않음)
# ============================================================
def detect_hot_region(thermal: np.ndarray):
    """배경보다 따뜻한 연결 영역 중 가장 큰 영역과 형태 정보를 반환함."""
    background_temp = float(np.median(thermal))
    threshold_temp = background_temp + TEMP_OFFSET
    core_threshold_temp = background_temp + CORE_TEMP_OFFSET
    mask = np.where(thermal >= threshold_temp, 255, 0).astype(np.uint8)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)
    if count <= 1:
        return None, background_temp, threshold_temp

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pixel_area = int(stats[largest_label, cv2.CC_STAT_AREA])
    if pixel_area < MIN_HOT_PIXELS:
        return None, background_temp, threshold_temp

    region_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, background_temp, threshold_temp

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = float(4.0 * np.pi * contour_area / (perimeter * perimeter)) if perimeter > 0.0 else 0.0
    aspect_ratio = float(w / h) if h > 0 else 0.0
    fill_ratio = float(pixel_area / (w * h)) if w > 0 and h > 0 else 0.0

    detection = {
        "contour": contour,
        "box": (int(x), int(y), int(w), int(h)),
        "circle": (float(circle_x), float(circle_y), float(radius)),
        "pixel_area": pixel_area,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
        "core_center_offset_ratio": None,
    }

    core_mask = np.where(thermal >= core_threshold_temp, 255, 0).astype(np.uint8)
    core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel)
    core_count, core_labels, _, core_centroids = cv2.connectedComponentsWithStats(core_mask, connectivity=8)

    best_core_label = 0
    best_overlap = 0
    for core_label in range(1, core_count):
        overlap = int(np.count_nonzero((core_labels == core_label) & (region_mask > 0)))
        if overlap > best_overlap:
            best_overlap = overlap
            best_core_label = core_label

    if best_core_label > 0:
        core_center = (
            float(core_centroids[best_core_label][0]),
            float(core_centroids[best_core_label][1]),
        )
        center_distance = float(np.hypot(core_center[0] - circle_x, core_center[1] - circle_y))
        detection["core_center_offset_ratio"] = center_distance / max(radius, 0.001)

    return detection, background_temp, threshold_temp


def is_head_shape(detection) -> bool:
    """가장 큰 발열 영역이 실험용 구 형태 조건을 만족하는지 확인함."""
    if detection is None:
        return False

    core_offset_ratio = detection["core_center_offset_ratio"]
    core_is_aligned = core_offset_ratio is None or core_offset_ratio <= MAX_CORE_CENTER_OFFSET_RATIO
    return (
        detection["circularity"] >= MIN_CIRCULARITY
        and MIN_ASPECT_RATIO <= detection["aspect_ratio"] <= MAX_ASPECT_RATIO
        and detection["fill_ratio"] >= MIN_FILL_RATIO
        and core_is_aligned
    )


def create_absolute_colormap(thermal: np.ndarray) -> np.ndarray:
    """24~36°C 고정 범위의 INFERNO 컬러맵을 생성함."""
    clipped = np.clip(thermal, TEMP_MIN, TEMP_MAX)
    image_8bit = ((clipped - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) * 255.0).astype(np.uint8)
    enlarged = cv2.resize(image_8bit, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(enlarged, cv2.COLORMAP_INFERNO)


def draw_detection(image: np.ndarray, detection, instant_match: bool, confirmed: bool) -> None:
    """발열 외곽선과 구 판정 결과를 출력 영상에 표시함 (image를 제자리에서 수정)."""
    if detection is None:
        return

    scale_x = DISPLAY_WIDTH / WIDTH
    scale_y = DISPLAY_HEIGHT / HEIGHT
    contour = detection["contour"].astype(np.float32)
    contour[:, :, 0] *= scale_x
    contour[:, :, 1] *= scale_y
    contour = np.rint(contour).astype(np.int32)

    x, y, w, h = detection["box"]
    box_start = (int(x * scale_x), int(y * scale_y))
    box_end = (int((x + w) * scale_x), int((y + h) * scale_y))
    circle_x, circle_y, radius = detection["circle"]
    circle_center = (int(circle_x * scale_x), int(circle_y * scale_y))
    circle_radius = int(radius * (scale_x + scale_y) / 2.0)

    color = (0, 255, 0) if confirmed else (0, 255, 255) if instant_match else (255, 255, 255)
    label = "person_head" if confirmed else "head shape" if instant_match else "hot region"
    cv2.drawContours(image, [contour], -1, color, 2)
    cv2.rectangle(image, box_start, box_end, color, 2)
    cv2.circle(image, circle_center, max(1, circle_radius), color, 2)
    cv2.putText(
        image, label, (box_start[0], max(20, box_start[1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
    )


# ============================================================
# 서보 팬 보정 (arda-thermal-test/servo_link.py를 그대로 이식)
# ============================================================
def offset_from_circle_x(circle_x: float) -> float:
    """detection["circle"][0](열 위치, 0~WIDTH-1)을 -1.0~1.0 정규화 편차로 변환."""
    half_width = (WIDTH - 1) / 2.0
    normalized = (circle_x - half_width) / half_width
    return max(-1.0, min(1.0, normalized))


def offset_from_circle_y(circle_y: float) -> float:
    """detection["circle"][1](행 위치, 0~HEIGHT-1)을 -1.0(화면 위)~1.0(화면 아래)로 변환.

    arda-servo가 카메라 설치 높이/기울기와 함께 z=0 평면(지면/수면) 기준
    거리를 역산할 때 쓴다."""
    half_height = (HEIGHT - 1) / 2.0
    normalized = (circle_y - half_height) / half_height
    return max(-1.0, min(1.0, normalized))


class ServoPanSender:
    """열화상 발열 방향 보정 UDP 송신기 (arda-servo/thermal_receiver.py 수신 규격)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9996):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, offset: float, vertical_offset: float | None = None, matched: bool = False) -> None:
        """matched: 이 프레임이 원하는 모양과 매칭됐는지 — arda-servo는 이
        값이 True인 프레임에서만 제어권(_thermal_engaged)을 넘겨받는다.
        보정 자체는 matched 여부와 무관하게 열이 감지된 모든 프레임에서
        보낸다(서보가 열원 방향을 계속 향하도록)."""
        payload = {"offset": float(offset), "ts": time.time(), "matched": bool(matched)}
        if vertical_offset is not None:
            payload["vertical_offset"] = float(vertical_offset)
        self._sock.sendto(json.dumps(payload).encode("utf-8"), self._addr)

    def give_up(self) -> None:
        payload = json.dumps({"give_up": True, "ts": time.time()}).encode("utf-8")
        self._sock.sendto(payload, self._addr)

    def confirm(self, vertical_offset: float | None = None) -> None:
        payload = {"confirmed": True, "ts": time.time()}
        if vertical_offset is not None:
            payload["vertical_offset"] = float(vertical_offset)
        self._sock.sendto(json.dumps(payload).encode("utf-8"), self._addr)

    def close(self) -> None:
        self._sock.close()


# ============================================================
# 레이더 트리거/판정 회신 UDP
# ============================================================
def _try_recv_trigger(sock: socket.socket) -> dict | None:
    """트리거 소켓에서 논블로킹으로 1건 시도. 없거나 파싱 실패면 None."""
    try:
        data, _ = sock.recvfrom(4096)
    except socket.timeout:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def wait_for_trigger(sock: socket.socket) -> dict:
    """새 트리거 1건을 받을 때까지 짧은 타임아웃으로 반복 폴링함."""
    while True:
        trigger = _try_recv_trigger(sock)
        if trigger is not None:
            return trigger


def send_verdict(sock: socket.socket, addr: tuple, person: bool) -> None:
    payload = json.dumps({"person": person, "ts": time.time()}).encode("utf-8")
    sock.sendto(payload, addr)


def send_engaged(sock: socket.socket, addr: tuple) -> None:
    """이번 관찰에서 원하는 모양과 처음 매칭된(matched=True) 순간 1회
    arda-radar로 보낸다.

    arda-radar는 이 신호를 받으면(판정 회신 전까지) 지금 대기 중인 낙하를
    confidence가 더 높은 새 후보로 선점하지 않는다 — 열화상이 원하는 모양과
    매칭되기 시작한 순간부터는 열화상이 우선권을 갖는다는 원칙을, 서보
    (arda_servo.ServoController._thermal_engaged)뿐 아니라 레이더 쪽 판정
    대기 상태에도 적용하기 위함이다. 신호가 없으면(구버전 thermal-camera 등)
    arda-radar는 기존처럼 confidence만으로 선점한다."""
    payload = json.dumps({"engaged": True, "ts": time.time()}).encode("utf-8")
    sock.sendto(payload, addr)


# ============================================================
# 관찰 상태 기계 (arda-thermal-test/main.py의 run_observation과 동일한 규칙)
# ============================================================
def run_observation(
    read_frame_fn,
    dwell_seconds: float,
    required_matches: int,
    display: bool,
    servo_sender: ServoPanSender | None = None,
    settle_offset: float = 0.15,
    listen_sock: socket.socket | None = None,
    reply_sock: socket.socket | None = None,
    reply_addr: tuple | None = None,
) -> tuple[bool | None, dict | None]:
    """사람 모양 발열 영역이 이번 관찰(dwell) 동안 누적으로 required_matches
    번 잡히면 즉시 True를 반환한다 — 연속일 필요는 없다. 중간에 매칭이
    실패한 프레임이 끼어도 누적 횟수는 줄어들지 않는다(예: matched=True가
    2번, False가 여러 번, 다시 matched=True가 1번 나오면 합쳐서 3회로
    카운트).

    servo_sender가 주어지면, 발열 영역이 검출될 때마다(사람 모양 확정 여부와
    무관하게) 프레임 중심 대비 좌우/세로 위치를 arda-servo로 전송해 관찰 중에도
    서보가 그 방향을 계속 따라가도록 한다.

    "포기하고 그만둘까"를 결정하는 dwell_seconds 카운트다운은 서보가 더 이상
    움직이지 않아도 될 때(발열 영역이 없거나, 있어도 프레임 중심의
    settle_offset 이내로 들어와 더 보낼 보정이 없을 때)에만 진행된다. 카운트다운이
    도는 중에 열원이 다시 움직여 서보가 쫓아가야 하면 그 즉시 카운트다운을
    취소하고, 서보가 새 위치로 이동을 마쳐 다시 안정되는 순간부터 dwell_seconds짜리
    카운트다운을 처음부터 새로 시작한다.

    매칭 자체는 서보 이동 여부와 무관하게 매 프레임 계속 진행하고, 매칭 성공
    시 누적 카운트는 이동 여부와 무관하게 늘어난다(리셋 로직 자체가 없음).

    listen_sock이 주어지면 매 프레임 이 소켓도 논블로킹으로 확인해, 새 트리거가
    도착하면(=arda-radar가 더 높은 확률의 낙하 후보를 확정했다는 뜻) 지금 하던
    관찰을 판정 없이 즉시 버리고 그 트리거로 재시작한다 — 단, 이번 관찰에서
    원하는 모양과 이미 한 번이라도 매칭돼(matched=True, required_matches
    중 1/N째) engaged 상태가 됐다면 새 트리거를 무시하고 지금 추적을 계속한다.
    단순히 배경보다 뜨거운 영역이 있다는 것만으로는(detection is not None이지만
    is_head_shape()가 False) engaged로 치지 않는다 — 사람 모양이 아닌 열원
    (반사광, 손 등)에 선점권을 뺏기지 않기 위함이다. 레이더 좌표는 대략적인
    초기 조준일 뿐이고, 열화상이 원하는 모양과 매칭되기 시작한 순간부터는
    열화상이 우선권을 갖는다는 원칙(arda-servo의 ServoController._thermal_engaged와
    동일)을 여기서도 지킨다. 반환값은 (결과, 대체 트리거) 튜플이다: 정상 종료 시
    (True/False, None), 새 트리거로 대체된 경우 (None, 그 트리거의 파싱된 딕셔너리)."""
    match_count = 0
    frame_number = 0
    give_up_deadline = None  # None이면 서보가 아직 이동 중 — 카운트다운 보류
    engaged = False  # 이번 관찰에서 원하는 모양과 한 번이라도 매칭됐는지(matched=True)

    while give_up_deadline is None or time.monotonic() < give_up_deadline:
        if listen_sock is not None:
            new_trigger = _try_recv_trigger(listen_sock)
            if new_trigger is not None:
                if engaged:
                    print("[제어권 유지] 더 높은 확률의 낙하 후보 트리거 수신 — 이미 매칭된 대상을 추적 중이라 무시하고 계속 관찰합니다")
                else:
                    print("[제어권 이동] 더 높은 확률의 낙하 후보 트리거 수신 — 현재 관찰을 중단하고 즉시 재시작합니다")
                    return None, new_trigger

        thermal = read_frame(read_frame_fn)
        if thermal is None:
            continue
        frame_number += 1

        detection, _background_temp, _threshold_temp = detect_hot_region(thermal)
        matched = is_head_shape(detection)  # 서보 이동 중에도 항상 시도함
        if matched and not engaged:
            # 원하는 모양과 "처음" 매칭된 순간(=match_count가 1이 되는 시점)부터
            # engaged로 친다 — 그냥 배경보다 뜨거운 영역이 있다는 것만으로는
            # (detection is not None) engaged로 치지 않는다.
            engaged = True
            if reply_sock is not None and reply_addr is not None:
                send_engaged(reply_sock, reply_addr)

        offset = None
        moving = False
        if detection is not None and servo_sender is not None:
            offset = offset_from_circle_x(detection["circle"][0])
            vertical_offset = offset_from_circle_y(detection["circle"][1])
            servo_sender.send(offset, vertical_offset, matched=matched)
            moving = abs(offset) > settle_offset

        if moving:
            give_up_deadline = None
        elif give_up_deadline is None:
            give_up_deadline = time.monotonic() + dwell_seconds

        if matched:
            match_count += 1

        confirmed = matched and match_count >= required_matches

        print(f"[열화상 매칭시도 {frame_number}] matched={'O' if matched else 'X'} "
              f"({match_count}/{required_matches}) confirmed={confirmed}")

        if display:
            image = create_absolute_colormap(thermal)
            draw_detection(image, detection, matched, confirmed)
            cv2.imshow(WINDOW_NAME, image)
            cv2.waitKey(1)

        if match_count >= required_matches:
            if servo_sender is not None:
                vertical_offset = offset_from_circle_y(detection["circle"][1])
                servo_sender.confirm(vertical_offset)
                print("[servo] 사람 확정 — arda-servo 로그에서 레이더 원좌표와 "
                      "추적 후 좌표를 확인하세요")
            return True, None

        time.sleep(FRAME_INTERVAL_S)

    if servo_sender is not None:
        servo_sender.give_up()
        print("[servo] 관찰 실패 — 추적 포기 신호 전송")

    return False, None


# ============================================================
# 진입점
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="열화상 person 확정 메인 흐름 (온도 임계값 기반)")
    parser.add_argument("--simulate", action="store_true", help="센서 없이 임의 배경 프레임으로 실행")
    parser.add_argument("--listen-host", default="0.0.0.0", help="레이더 트리거 수신 주소")
    parser.add_argument("--listen-port", type=int, default=9998, help="레이더 트리거 수신 포트")
    parser.add_argument("--reply-host", default="127.0.0.1", help="판정 결과 회신 주소 (레이더)")
    parser.add_argument("--reply-port", type=int, default=9997, help="판정 결과 회신 포트 (레이더)")
    parser.add_argument("--dwell-seconds", type=float, default=10.0, help="트리거 후 최대 관찰 시간(초)")
    parser.add_argument("--required-matches", type=int, default=3, help="확정에 필요한 누적 매칭 횟수(연속일 필요 없음)")
    parser.add_argument(
        "--settle-offset", type=float, default=0.15,
        help="서보 전송 시, 발열 위치가 프레임 중심에서 이 값(정규화 -1.0~1.0) 이내일 때만 "
             "'서보 이동이 끝났다'고 보고 연속 매칭 카운트를 진행함. --no-servo-out이면 적용되지 않음",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="열화상 컬러맵 창을 띄우지 않음 (기본: DISPLAY 환경변수가 있으면 자동으로 띄움)",
    )
    parser.add_argument(
        "--no-servo-out", action="store_true",
        help="서보 제어기(arda-servo)로의 발열 방향 보정 UDP 전송 비활성화",
    )
    parser.add_argument("--servo-host", default="127.0.0.1", help="서보 제어기(arda-servo) UDP 호스트")
    parser.add_argument("--servo-port", type=int, default=9996, help="서보 제어기(arda-servo) UDP 포트")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    display = not args.no_viz and bool(os.environ.get("DISPLAY"))
    if not args.no_viz and not display:
        print("[display] DISPLAY 환경변수가 없어 컬러맵 창 없이 실행합니다")

    i2c = None
    try:
        i2c, read_frame_fn = initialize_sensor(simulate=args.simulate)
    except RuntimeError as exc:
        print(f"[실행 오류] {exc}", file=sys.stderr)
        return 1

    servo_sender = None if args.no_servo_out else ServoPanSender(args.servo_host, args.servo_port)
    if servo_sender:
        print(f"[servo] 발열 방향 보정 전송 활성화 — UDP {args.servo_host}:{args.servo_port}")

    print(f"[warmup] 센서 안정화 {SENSOR_WARMUP_FRAMES}프레임 대기 중...")
    for _ in range(SENSOR_WARMUP_FRAMES):
        read_frame(read_frame_fn)
        time.sleep(FRAME_INTERVAL_S)

    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind((args.listen_host, args.listen_port))
    listen_sock.settimeout(0.05)

    reply_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reply_addr = (args.reply_host, args.reply_port)

    print(f"[대기] {args.listen_host}:{args.listen_port} 에서 레이더 트리거 대기 중 "
          f"(dwell={args.dwell_seconds}s, 회신={args.reply_host}:{args.reply_port})")

    try:
        trigger = wait_for_trigger(listen_sock)
        while True:
            sent_ts = trigger.get("ts")
            latency_ms = (time.time() - sent_ts) * 1000 if sent_ts else None
            latency_note = f" (전송 후 {latency_ms:.0f}ms)" if latency_ms is not None else ""
            print(f"[트리거 수신]{latency_note} — 최대 {args.dwell_seconds}s 관찰 시작")
            person, superseding_trigger = run_observation(
                read_frame_fn, args.dwell_seconds, args.required_matches, display,
                servo_sender, args.settle_offset, listen_sock, reply_sock, reply_addr,
            )
            if person is None:
                trigger = superseding_trigger
                continue

            print(f"[판정 완료] person={person}  ({'사람 확인됨' if person else '사람 아님/미확인'})")
            send_verdict(reply_sock, reply_addr, person)
            trigger = wait_for_trigger(listen_sock)

    except KeyboardInterrupt:
        print("사용자 중단")
    finally:
        listen_sock.close()
        reply_sock.close()
        if servo_sender:
            servo_sender.close()
        if display:
            cv2.destroyAllWindows()
        if i2c is not None and hasattr(i2c, "deinit"):
            i2c.deinit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
