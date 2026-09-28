"""Jetson GPIO 기반 팬(pan) 서보 제어."""

from .utils import get_logger

logger = get_logger(__name__)

try:
    import Jetson.GPIO as GPIO
    _HAS_GPIO = True
except Exception as e:  # noqa: BLE001 — Jetson.GPIO는 미지원 보드에서 import 시점에 일반 Exception을 던짐
    GPIO = None
    _HAS_GPIO = False
    _GPIO_IMPORT_ERROR = e

PWM_FREQ_HZ = 50.0    # 표준 아날로그 서보 PWM 주파수
MIN_PULSE_MS = 0.5    # min_deg에 대응하는 펄스폭
MAX_PULSE_MS = 2.5    # max_deg에 대응하는 펄스폭


def _board_pin_to_tegra_soc(board_pin: int) -> str:
    """BOARD(물리 핀 번호)를 TEGRA_SOC 핀 이름으로 변환.

    arda-raset처럼 서보와 thermal-camera(Adafruit Blinka, `import board`)를
    한 프로세스에서 같이 쓰는 경우를 위한 것 — Blinka는 내부적으로
    GPIO.setmode(GPIO.TEGRA_SOC)를 호출하는데, Jetson.GPIO는 프로세스당
    setmode 모드를 하나로 고정하고 다른 모드로 재호출하면 예외를 던진다
    (같은 모드 재호출은 허용). 서보도 TEGRA_SOC로 맞춰두면 어느 쪽이
    먼저 초기화되든 충돌하지 않는다. 설정 파일은 배선 편의를 위해 BOARD
    물리 핀 번호를 그대로 쓰고, 변환은 여기서만 한다.
    """
    from Jetson.GPIO import gpio_pin_data

    _, _, channel_data = gpio_pin_data.get_data()
    line_offset = channel_data["BOARD"][board_pin].line_offset
    for name, info in channel_data["TEGRA_SOC"].items():
        if info.line_offset == line_offset:
            return name
    raise ValueError(f"BOARD 핀 {board_pin}에 대응하는 TEGRA_SOC 핀을 찾지 못했습니다")


class PanServo:
    """단일 축(pan) 서보 모터 제어기. Jetson.GPIO 소프트웨어 PWM 사용.

    Jetson.GPIO를 로드할 수 없는 환경(보드 미인식, 권한 부족 등)에서는
    자동으로 시뮬레이션 모드로 동작해 상위 로직(좌표 수신·각도 계산)을
    하드웨어 없이도 개발/테스트할 수 있게 한다.
    """

    def __init__(
        self,
        pin: int,
        min_deg: float = 0.0,
        max_deg: float = 180.0,
        min_pulse_ms: float = MIN_PULSE_MS,
        max_pulse_ms: float = MAX_PULSE_MS,
        simulate: bool = False,
    ):
        self._pin = pin
        self._min_deg = min_deg
        self._max_deg = max_deg
        self._min_pulse_ms = min_pulse_ms
        self._max_pulse_ms = max_pulse_ms
        self._angle: float | None = None
        self._pwm = None
        self._channel: str | None = None

        self._simulate = simulate or not _HAS_GPIO
        if not _HAS_GPIO and not simulate:
            logger.warning("Jetson.GPIO 로드 실패(%s) — 시뮬레이션 모드로 전환", _GPIO_IMPORT_ERROR)
        if self._simulate:
            if simulate:
                logger.info("시뮬레이션 모드로 실행 — 실제 서보는 움직이지 않음")
            return

        self._channel = _board_pin_to_tegra_soc(self._pin)
        GPIO.setmode(GPIO.TEGRA_SOC)
        GPIO.setup(self._channel, GPIO.OUT)
        self._pwm = GPIO.PWM(self._channel, PWM_FREQ_HZ)
        self._pwm.start(0)

    @property
    def min_deg(self) -> float:
        return self._min_deg

    @property
    def max_deg(self) -> float:
        return self._max_deg

    @property
    def angle(self) -> float | None:
        return self._angle

    def set_angle(self, angle_deg: float) -> None:
        angle_deg = max(self._min_deg, min(self._max_deg, angle_deg))
        self._angle = angle_deg

        if self._simulate:
            logger.debug("SIM set_angle=%.1f°", angle_deg)
            return

        span = self._max_deg - self._min_deg
        pulse_ms = self._min_pulse_ms + (angle_deg - self._min_deg) / span * (
            self._max_pulse_ms - self._min_pulse_ms
        )
        duty = pulse_ms / (1000.0 / PWM_FREQ_HZ) * 100.0
        self._pwm.ChangeDutyCycle(duty)

    def close(self) -> None:
        if self._simulate:
            return
        self._pwm.stop()
        GPIO.cleanup(self._channel)
