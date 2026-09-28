"""설치 위치(위도/경도)와 정면 방위각 기준 실좌표(GPS) 변환.

arda-radar(arda/utils/site.py)의 local_to_latlon()과 동일한 수식이다.
arda-servo는 arda-radar와 완전히 분리된 프로젝트라 코드를 직접 참조하지
않고 이 파일에 그대로 복사해서 쓴다 — config/settings.yaml의 site.lat/lon/
heading_deg 값도 arda-radar 쪽 site 설정과 같은 값으로 수동으로 맞춰야
한다.
"""

import math

# 위도 1도당 거리(m). 지구를 구로 근사한 값 — 위도에 따라 실제로는 미세하게
# 다르지만(적도 110.57km ~ 극지 111.69km) 이 프로젝트의 탐지 범위(수 m)에서는
# 그 차이가 무시할 수준이라 하나의 상수로 충분하다.
METERS_PER_DEG_LAT = 111_320.0


def local_to_latlon(
    x: float,
    y: float,
    site_lat: float,
    site_lon: float,
    heading_deg: float = 0.0,
) -> tuple[float, float]:
    """센서 기준 로컬 좌표(X: 좌우 +우측, Y: 정면 거리, 단위 m)를 설치 지점의
    위도/경도와 정면 방위각(heading_deg)을 이용해 실제 위도/경도로 변환한다.

    heading_deg는 센서가 정면(Y축, boresight)으로 바라보는 나침반 방위각이다
    (정북 0°, 시계방향으로 증가 — 동 90°, 남 180°, 서 270°). 로컬 좌표를 이
    각도만큼 회전시켜 동/북 방향 변위로 바꾸므로, 기기가 어느 방향을 보고
    설치되든(정북이 아니어도) 그 방향에 맞는 부호·축으로 자동 계산된다.

    arda-servo에서는 레이더 원좌표(coord.x, coord.y)와 pan_angle_to_xyz()로
    역산한 열화상 추적 후 좌표 모두 "레이더 로컬 좌표계" 기준이므로, 이
    함수에는 항상 레이더 쪽 site.lat/lon/heading_deg를 그대로 넣는다 — 서보
    자체의 mount_offset은 xyz_to_pan_angle()/pan_angle_to_xyz()에서 이미
    보정된 뒤이므로 여기서 다시 반영할 필요가 없다.
    """
    heading_rad = math.radians(heading_deg)

    # 방위각 θ 방향의 단위 벡터는 (East=sin θ, North=cos θ). 정면(Y)의 방위각은
    # heading_deg 그 자체, 우측(X)은 그보다 시계방향으로 90도 회전한 방위각이다.
    east_m = x * math.cos(heading_rad) + y * math.sin(heading_rad)
    north_m = -x * math.sin(heading_rad) + y * math.cos(heading_rad)

    delta_lat = north_m / METERS_PER_DEG_LAT
    delta_lon = east_m / (METERS_PER_DEG_LAT * math.cos(math.radians(site_lat)))

    return site_lat + delta_lat, site_lon + delta_lon
