"""레이더-서보 연동용 person 확정 메인 흐름 (YOLO 커스텀 모델 기반 판정).

thermal_main.py와 통신 프로토콜/관찰 상태 기계는 완전히 동일하고, 판정
방식만 다르다 — 열화상 컬러맵 전체 프레임을 Roboflow YOLO26 모델
(models/s_yolo26.pt, 단일 클래스 "sphere")에 직접 넣는다. 레이더 트리거 뒤
최근 5프레임 중 3프레임 이상에서 sphere가 검출되면 사람 머리로 확정한다.

라즈베리파이는 requirements.txt에 이미 ultralytics가 포함돼 있어 별도 설치
없이 --device cpu로 바로 쓸 수 있다. Jetson에서 CUDA로 돌리려면
requirements-jetson.txt 설치 후 requirements-yolo-jetson.txt / fix_cuda_cudss.sh를
추가로 참고 (일반 PyPI torch는 이 보드에서 CUDA를 못 잡는다). README 23장 참고.
"""

import argparse
import json
import os
import socket
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# 센서/화면 설정값 (thermal_main.py와 동일)
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
# (read_frame() 참고). thermal_main.py의 동일 상수와 반드시 같이 맞출 것.
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
# 이 보정은 센서 칩 자체의 배선 특성이라 ROTATE_90_CCW(하우징 장착 방향)와
# 무관하게 원본 좌표계에서 먼저 적용한다.
FLIP_HORIZONTAL = True

# thermal_recorder.py의 학습 영상 전처리와 동일한 절대 온도 컬러맵 범위
TEMP_MIN = 26.0
TEMP_MAX = 38.0

SENSOR_WARMUP_FRAMES = 5
FRAME_INTERVAL_S = 0.5  # MLX90640 REFRESH_2_HZ 기준
WINDOW_NAME = "ARDA Thermal Main (YOLO) — 관찰 중"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "t1_ver5.pt"
TARGET_CLASS_NAME = "sphere"
CONFIDENCE_THRESHOLD = 0.25
DECISION_WINDOW_FRAMES = 5
REQUIRED_DETECTION_FRAMES = 3


# ============================================================
# 센서
# ============================================================
def initialize_sensor(simulate: bool = False):
    """I2C 버스와 MLX90640 센서를 초기화함. simulate=True면 하드웨어 없이
    배경(24도)+잡음만 있는 임의 프레임을 반환하는 폴백 함수를 대신 돌려준다."""
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


def create_absolute_colormap(thermal: np.ndarray) -> np.ndarray:
    """thermal_recorder.py와 같은 26~38°C 고정 INFERNO 컬러맵을 생성함."""
    clipped = np.clip(thermal, TEMP_MIN, TEMP_MAX)
    image_8bit = ((clipped - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) * 255.0).astype(np.uint8)
    enlarged = cv2.resize(image_8bit, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(enlarged, cv2.COLORMAP_INFERNO)


# ============================================================
# YOLO 입력 전용 배경-상대 컬러맵 (화면 표시용 create_absolute_colormap()과는
# 별개 — 그쪽은 실제 온도를 그대로 보여줘야 하므로 이 보정을 안 받는다)
# ============================================================
# 사람 표면온도는 배경과 무관하게 대략 고정값(36°C)으로 본다 — 배경이
# 더워진다고 사람이 같이 더 뜨거워지는 게 아니라, 사람-배경 온도차(대비)만
# 배경에 따라 줄어들 뿐이다. 그래서 컬러맵 상단은 항상 36°C로 고정하고,
# 하단만 그 프레임의 배경(중앙값)에 맞춰 움직인다 — 더운 날처럼 실제
# 온도차가 좁아지는 상황에서도, 남아있는 그 차이를 0~255 전체로 늘려 펴서
# 색 대비를 항상 최대로 유지한다. 배경이 사람 온도(36°C)에 바짝 붙거나
# 넘어서는 극단(폭염 등)에 대비해 최소 대비(MIN_CONTRAST_MARGIN)와 전체
# 하한(ABSOLUTE_TEMP_FLOOR, 10°C)을 둔다.
PERSON_FIXED_TEMP = 36.0
ABSOLUTE_TEMP_FLOOR = 10.0
MIN_CONTRAST_MARGIN = 2.0


def create_relative_colormap(thermal: np.ndarray) -> np.ndarray:
    """사람 온도(36°C 고정) 대비 배경(그 프레임의 중앙값) 온도차를 항상
    0~255 전체 범위로 늘려 펴서 보여주는 컬러맵 — YOLO 탐지 입력 전용.

    상단은 항상 PERSON_FIXED_TEMP(36°C)로 고정하고, 하단만 배경 온도를
    따라간다(단, ABSOLUTE_TEMP_FLOOR 밑으로도, 상단과 MIN_CONTRAST_MARGIN
    미만으로 가깝게도 안 내려가게 막는다). 배경이 25°C든 30°C든 사람은
    항상 색 범위 최상단으로 매핑된다 — 실제로 41°C가 될 필요가 없다.
    """
    background_temp = float(np.median(thermal))
    temp_max = PERSON_FIXED_TEMP
    temp_min = min(background_temp, temp_max - MIN_CONTRAST_MARGIN)
    temp_min = max(ABSOLUTE_TEMP_FLOOR, temp_min)
    clipped = np.clip(thermal, temp_min, temp_max)
    image_8bit = ((clipped - temp_min) / (temp_max - temp_min) * 255.0).astype(np.uint8)
    enlarged = cv2.resize(image_8bit, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap(enlarged, cv2.COLORMAP_INFERNO)


# ============================================================
# YOLO 커스텀 모델 판정 (arda-thermal-test/yolo_detection.py를 그대로 이식)
# ============================================================
def load_yolo_model(model_path: Path = DEFAULT_MODEL_PATH, device: str = "cuda"):
    """Ultralytics로 커스텀 모델을 불러옴. (model, 실제 사용할 device) 튜플을 반환한다.

    device="cuda"가 기본값이다 — 일반 PyPI torch는 이 Jetson(L4T)에서 CUDA를
    못 잡지만, jetson-ai-lab 인덱스의 torch(requirements-yolo-jetson.txt 참고)는
    CUDA가 정상 동작한다. GPU를 못 쓰는 환경이면 자동으로 CPU로 대체한다."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics/torch가 설치되어 있지 않습니다. 라즈베리파이는 requirements.txt에 "
            "이미 ultralytics가 포함돼 있으니 --device cpu로 실행하세요. Jetson은 "
            "'pip install -r requirements-yolo-jetson.txt' 실행 후 ./fix_cuda_cudss.sh도 실행하세요."
        ) from exc

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO 모델 파일을 찾을 수 없습니다: {model_path}\n"
            "ARDA-2026/thermal_human_yolo에서 학습한 .pt 파일을 이 경로에 준비하세요."
        )

    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        raise RuntimeError(f"YOLO 모델을 불러오지 못했습니다: {model_path}\n원인: {exc}") from exc

    names = model.names
    class_names = set(names.values()) if isinstance(names, dict) else set(names)
    if TARGET_CLASS_NAME not in class_names:
        raise RuntimeError(
            f"모델 클래스가 예상과 다릅니다: {names}. "
            f"'{TARGET_CLASS_NAME}' 클래스가 필요합니다."
        )

    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                print("[yolo] CUDA를 사용할 수 없어 CPU로 대체합니다 (느릴 수 있음)")
                device = "cpu"
        except ImportError:
            device = "cpu"

    return model, device


def detect_person_candidates(
    color_image: np.ndarray,
    model,
    device: str = "cuda",
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """열화상 컬러맵 전체 프레임에 YOLO를 돌려 검출된 후보 목록을 반환함.

    Roboflow 모델에서 sphere로 검출된 박스만 후보로 본다. 좌표는
    color_image 픽셀 기준
    (DISPLAY_WIDTH x DISPLAY_HEIGHT)이다."""
    results = model.predict(source=color_image, conf=confidence_threshold, device=device, verbose=False)

    candidates = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = str(result.names.get(class_id, class_id))
            if class_name != TARGET_CLASS_NAME:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            candidates.append(
                {
                    "box": (x1, y1, x2, y2),
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "confidence": float(box.conf[0]),
                    "class_name": class_name,
                }
            )
    return candidates


def best_candidate(candidates: list[dict]) -> dict | None:
    """가장 신뢰도 높은 후보 하나를 반환함 (없으면 None)."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: c["confidence"])


def candidate_to_grid_xy(candidate: dict) -> tuple[float, float]:
    """YOLO 후보 중심 픽셀(DISPLAY_WIDTH x DISPLAY_HEIGHT 기준)을 열화상 원본
    32x24 격자 좌표로 변환함 — offset_from_circle_x/y가 이 격자 좌표를 입력으로
    받으므로, 이 변환을 거치면 thermal_main.py 경로와 동일한 서보 추적 로직을
    그대로 재사용할 수 있다."""
    cx, cy = candidate["center"]
    return cx / (DISPLAY_WIDTH / WIDTH), cy / (DISPLAY_HEIGHT / HEIGHT)


def draw_person_candidates(image: np.ndarray, candidates: list[dict], best: dict | None, confirmed: bool = False) -> None:
    """검출된 모든 후보를 표시함 (image를 제자리에서 수정).

    확정(초록) > 추적 중인 최우선 후보(노랑) > 그 외 후보(흰색) 3단계 색 규칙은
    thermal_main.py의 draw_detection()과 동일하다."""
    for item in candidates:
        x1, y1, x2, y2 = (int(v) for v in item["box"])
        is_best = best is not None and item is best
        if is_best and confirmed:
            color = (0, 255, 0)
            label = f"PERSON HEAD {item['confidence']:.2f}"
        elif is_best:
            color = (0, 255, 255)
            label = f"sphere candidate {item['confidence']:.2f}"
        else:
            color = (255, 255, 255)
            label = f"sphere {item['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )


# ============================================================
# 서보 팬 보정 (arda-thermal-test/servo_link.py를 그대로 이식)
# ============================================================
def offset_from_circle_x(circle_x: float) -> float:
    half_width = (WIDTH - 1) / 2.0
    normalized = (circle_x - half_width) / half_width
    return max(-1.0, min(1.0, normalized))


def offset_from_circle_y(circle_y: float) -> float:
    half_height = (HEIGHT - 1) / 2.0
    normalized = (circle_y - half_height) / half_height
    return max(-1.0, min(1.0, normalized))


class ServoPanSender:
    """열화상 발열 방향 보정 UDP 송신기 (arda-servo/thermal_receiver.py 수신 규격)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9996):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, offset: float, vertical_offset: float | None = None, matched: bool = False) -> None:
        """matched: 이 프레임이 원하는 모양(sphere)과 매칭됐는지 — arda-servo는
        이 값이 True인 프레임에서만 제어권(_thermal_engaged)을 넘겨받는다.
        보정 자체는 matched 여부와 무관하게 후보가 검출된 모든 프레임에서
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
    try:
        data, _ = sock.recvfrom(4096)
    except socket.timeout:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def wait_for_trigger(sock: socket.socket) -> dict:
    while True:
        trigger = _try_recv_trigger(sock)
        if trigger is not None:
            return trigger


def send_verdict(sock: socket.socket, addr: tuple, person: bool) -> None:
    payload = json.dumps({"person": person, "ts": time.time()}).encode("utf-8")
    sock.sendto(payload, addr)


def send_engaged(sock: socket.socket, addr: tuple) -> None:
    """이번 관찰에서 원하는 모양(sphere 클래스)과 처음 매칭된 순간 1회
    arda-radar로 보낸다 — thermal_main.py의 send_engaged()와 동일한 목적
    (자세한 설명은 그쪽 참고). YOLO 백엔드는 모델이 이미 "sphere" 클래스로
    분류한 후보만 매칭으로 치므로(=matched와 사실상 동일 조건) 별도 임계값
    조정 없이도 threshold 백엔드와 같은 원칙이 성립한다."""
    payload = json.dumps({"engaged": True, "ts": time.time()}).encode("utf-8")
    sock.sendto(payload, addr)


# ============================================================
# 관찰 상태 기계 (thermal_main.py의 run_observation과 동일한 규칙,
# 판정만 YOLO 후보로 교체)
# ============================================================
def run_observation(
    read_frame_fn,
    model,
    device: str,
    confidence_threshold: float,
    dwell_seconds: float,
    decision_window_frames: int,
    required_detections: int,
    display: bool,
    servo_sender: ServoPanSender | None = None,
    settle_offset: float = 0.15,
    listen_sock: socket.socket | None = None,
    reply_sock: socket.socket | None = None,
    reply_addr: tuple | None = None,
) -> tuple[bool | None, dict | None]:
    """최근 decision_window_frames 중 required_detections 이상 sphere가
    검출되면 True를 반환함. 레이더 트리거와 판정 회신 구조는 유지함 — 단,
    이번 관찰에서 원하는 모양(sphere)과 이미 한 번이라도 매칭돼 engaged
    상태가 됐다면 새 트리거를 무시하고 지금 추적을 계속한다(자세한 원칙은
    thermal_main.py의 run_observation() 문서 참고)."""
    detection_history = deque(maxlen=decision_window_frames)
    frame_number = 0
    give_up_deadline = None
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

        color_image = create_absolute_colormap(thermal)
        yolo_input = create_relative_colormap(thermal)
        candidates = detect_person_candidates(yolo_input, model, device, confidence_threshold)
        best = best_candidate(candidates)
        matched = best is not None
        if matched and not engaged:
            engaged = True
            if reply_sock is not None and reply_addr is not None:
                send_engaged(reply_sock, reply_addr)

        offset = None
        moving = False
        if best is not None and servo_sender is not None:
            grid_x, grid_y = candidate_to_grid_xy(best)
            offset = offset_from_circle_x(grid_x)
            vertical_offset = offset_from_circle_y(grid_y)
            servo_sender.send(offset, vertical_offset, matched=matched)
            moving = abs(offset) > settle_offset

        if moving:
            give_up_deadline = None
        elif give_up_deadline is None:
            give_up_deadline = time.monotonic() + dwell_seconds

        detection_history.append(matched)
        detection_count = sum(detection_history)
        confirmed = (
            len(detection_history) == decision_window_frames
            and detection_count >= required_detections
        )

        conf_note = f" conf={best['confidence']:.2f}" if best is not None else ""
        print(f"[열화상 매칭시도 {frame_number}] matched={'O' if matched else 'X'}{conf_note} "
              f"({detection_count}/{decision_window_frames}) confirmed={confirmed}")

        if display:
            draw_person_candidates(color_image, candidates, best, confirmed)
            cv2.imshow(WINDOW_NAME, color_image)
            cv2.waitKey(1)

        if confirmed:
            if servo_sender is not None and best is not None:
                _, grid_y = candidate_to_grid_xy(best)
                vertical_offset = offset_from_circle_y(grid_y)
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
    parser = argparse.ArgumentParser(description="열화상 person 확정 메인 흐름 (YOLO 커스텀 모델 기반)")
    parser.add_argument("--simulate", action="store_true", help="센서 없이 임의 배경 프레임으로 실행")
    parser.add_argument("--listen-host", default="0.0.0.0", help="레이더 트리거 수신 주소")
    parser.add_argument("--listen-port", type=int, default=9998, help="레이더 트리거 수신 포트")
    parser.add_argument("--reply-host", default="127.0.0.1", help="판정 결과 회신 주소 (레이더)")
    parser.add_argument("--reply-port", type=int, default=9997, help="판정 결과 회신 포트 (레이더)")
    parser.add_argument("--dwell-seconds", type=float, default=10.0, help="트리거 후 최대 관찰 시간(초)")
    parser.add_argument(
        "--decision-window-frames",
        type=int,
        default=DECISION_WINDOW_FRAMES,
        help="사람 머리 판정에 사용할 최근 프레임 수(기본: 5)",
    )
    parser.add_argument(
        "--required-detections",
        type=int,
        default=REQUIRED_DETECTION_FRAMES,
        help="판정 창에서 필요한 sphere 검출 프레임 수(기본: 3)",
    )
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
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="YOLO 모델(.pt) 경로")
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD, help="YOLO 검출 신뢰도 임계값")
    parser.add_argument("--device", default="cuda", help="YOLO 추론 디바이스 ('cuda' 또는 'cpu')")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.decision_window_frames <= 0
        or args.required_detections <= 0
        or args.required_detections > args.decision_window_frames
    ):
        print(
            "[인자 오류] 0 < required-detections <= decision-window-frames 조건이 필요합니다.",
            file=sys.stderr,
        )
        return 2
    display = not args.no_viz and bool(os.environ.get("DISPLAY"))
    if not args.no_viz and not display:
        print("[display] DISPLAY 환경변수가 없어 컬러맵 창 없이 실행합니다")

    try:
        print(f"[yolo] 모델 로딩 중: {args.model_path} (device={args.device})")
        model, device = load_yolo_model(args.model_path, args.device)
        print(f"[yolo] 모델 로딩 완료 — classes={model.names}, device={device}")
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[초기화 오류] {exc}", file=sys.stderr)
        return 1

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
                read_frame_fn, model, device, args.confidence_threshold,
                args.dwell_seconds, args.decision_window_frames,
                args.required_detections, display,
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
