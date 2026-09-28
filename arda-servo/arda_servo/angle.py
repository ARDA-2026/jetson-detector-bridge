"""레이더 좌표 → 팬(pan) 서보 각도 변환."""

import math


def xyz_to_pan_angle(
    x: float,
    y: float,
    center_deg: float = 90.0,
    min_deg: float = 0.0,
    max_deg: float = 180.0,
    invert: bool = False,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> float:
    """레이더 좌표(X: 좌우, Y: 전방 거리, 단위 m)를 서보 각도(도)로 변환한다.

    azimuth = atan2(x, y) — 센서 정면(y축) 방향이 0도, 우측(+x)이 양수.
    center_deg를 정면 기준각으로 삼아 azimuth를 더한 뒤 가동 범위로 clamp한다.

    offset_x/offset_y(m)는 서보가 레이더 대비 설치된 위치 차이다 — 레이더
    좌표계 기준으로 "서보가 레이더에서 얼마나 떨어져 있는지"를 나타내며,
    타겟 좌표에서 이만큼을 뺀 뒤 azimuth를 계산해 서보 위치 기준으로
    보정한다. 두 장치가 같은 위치에 있으면 0(기본값)으로 둔다.
    """
    x -= offset_x
    y -= offset_y

    if y <= 0:
        y = 1e-6  # 서보 위치 기준으로 재계산한 뒤에도 방어적으로 clamp

    azimuth_deg = math.degrees(math.atan2(x, y))
    if invert:
        azimuth_deg = -azimuth_deg

    angle = center_deg + azimuth_deg
    return max(min_deg, min(max_deg, angle))


def pan_angle_to_xyz(
    angle_deg: float,
    range_m: float,
    center_deg: float = 90.0,
    invert: bool = False,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> tuple[float, float]:
    """서보 각도(도)로부터 레이더 좌표계 기준 (x, y)를 역산한다. xyz_to_pan_angle()의 역변환.

    각도만으로는 방향(azimuth)만 복원되고 거리 정보는 사라진다 — 원점에서
    같은 방향으로 뻗은 직선 위의 모든 (x, y)가 같은 각도를 만들기 때문이다.
    그래서 range_m(원점 기준 거리)을 함께 받아야 유일한 좌표가 정해진다.
    보통 이 dwell을 시작시킨 레이더 원좌표의 거리를 그대로 써서, "방향은
    열화상이 다듬은 값, 거리는 레이더가 처음 잰 값"으로 좌표를 재구성한다.

    xyz_to_pan_angle()이 각도를 [min_deg, max_deg]로 clamp하므로, 입력
    각도가 서보 가동 범위 끝에서 clamp된 상태였다면 이 역변환은 실제
    azimuth가 아니라 clamp된 각도 기준으로 좌표를 계산한다.
    """
    azimuth_deg = angle_deg - center_deg
    if invert:
        azimuth_deg = -azimuth_deg

    azimuth_rad = math.radians(azimuth_deg)
    x = offset_x + range_m * math.sin(azimuth_rad)
    y = offset_y + range_m * math.cos(azimuth_rad)
    return x, y


def elevation_range(
    vertical_offset: float,
    install_height_m: float,
    camera_tilt_deg: float,
    vertical_fov_deg: float,
) -> float:
    """열화상 프레임의 세로(행) 편차로부터 z=0 평면(지면/수면)까지의 수평 거리를 구한다.

    카메라가 install_height_m 높이에 고정 설치되어 수평 기준 아래로
    camera_tilt_deg만큼 기울어져 있고(팬만 좌우로 돌 뿐 상하로는 움직이지
    않음), 세로 화각이 vertical_fov_deg라고 가정한다.

    vertical_offset은 프레임 세로 중심 대비 편차(-1.0~1.0)로, -1.0이 화면
    맨 위(카메라 기준 앙각이 얕음 → z=0 평면과 더 먼 지점과 만남), +1.0이
    화면 맨 아래(앙각이 깊음 → 더 가까운 지점)이다:

        elevation_deg = camera_tilt_deg + vertical_offset * (vertical_fov_deg / 2)
        range_m = install_height_m / tan(elevation_deg)

    elevation_deg가 0 이하면(카메라가 수평이거나 그 위를 보는 셈이라 z=0
    평면과 만나지 않음) ValueError를 낸다 — 호출 측에서 대체 거리(레이더
    원거리 등)로 대신 처리해야 한다.
    """
    elevation_deg = camera_tilt_deg + vertical_offset * (vertical_fov_deg / 2.0)
    if elevation_deg <= 0:
        raise ValueError(
            f"elevation_deg={elevation_deg:.1f}° — 카메라가 수평 이상을 보고 있어 "
            "z=0 평면과 만나지 않음"
        )
    return install_height_m / math.tan(math.radians(elevation_deg))
