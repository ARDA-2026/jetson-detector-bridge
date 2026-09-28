# ARDA Jetson 실행 가이드

Jetson Orin Nano · JetPack 6.2 · Ubuntu 22.04 · NetworkManager 기준입니다.

아래 명령의 `~/ARDA-2026/jetson`은 현재 Jetson의 실제 경로입니다. 다른 위치에 clone했다면 그 경로로 바꾸세요.

> **AP 전환은 Jetson의 현재 Wi-Fi를 끊습니다.** `start_demo.sh`와 `stop_demo.sh`는 Jetson 로컬 터미널 또는 USB/유선 연결에서 실행하세요. 이 폴더를 구성하면서 실제 AP 전환은 실행하지 않았습니다.

## 1. Jetson이 하는 일

```text
레이더(IWR6843) ── 낙하 후보 ──┐
                               ├─ arda-raset ── POST /report ── Windows 노트북
열화상(MLX90640) ── 사람 판정 ─┤                         (hanriver.py:8000)
서보 ──────────── 방향 추적 ────┘
```

- `arda-raset`이 레이더, 열화상, 서보 코드를 **한 Python 프로세스**에서 실행합니다.
- 노트북은 `arda-algo_general`의 **`drone_control` 브랜치**에서 `hanriver.py`를 실행합니다. 노트북 코드는 이 폴더에 포함되지 않습니다.
- 참고 문서의 `arda-bringup`은 이 실행 경로에 필요하지 않습니다.
- 부팅 시에는 NetworkManager가 사용 가능한 Wi-Fi에 연결하고 센서 서비스가 시작됩니다. **AP는 부팅 시 자동으로 켜지지 않습니다.**
- 시연 시에만 `ARDA_NET` AP를 켭니다. Windows의 **“인터넷 없음”은 정상**입니다.

## 2. 폴더 구조

```text
jetson/
├── README.md
├── config/
│   ├── runtime.env          # 웹 주소, 설치 좌표, 장치 포트, 실행 옵션
│   ├── hostapd.conf         # ARDA_NET / WPA2
│   └── dnsmasq.conf         # Windows에 192.168.10.10~100 할당
├── scripts/
│   ├── run_sensor.sh        # arda-raset 실행
│   ├── set_report_url.sh    # Windows IP를 실행 중 서비스에 임시 적용
│   ├── install_service.sh   # 부팅 서비스 등록
│   ├── start_demo.sh        # 센서 서비스 확인 → AP 시작
│   ├── stop_demo.sh         # 센서 중지 → AP 종료 → Wi-Fi 복귀
│   ├── demo_start.sh        # AP 시작 자동화 복사본
│   └── demo_stop.sh         # AP 종료 자동화 복사본
├── services/arda-jetson.service.in
├── arda-raset/              # 통합 실행기, pyproject.toml, uv.lock
├── arda-radar/              # 레이더 코드, 설정, .cfg 프로필
├── arda-servo/              # 서보 코드와 GPIO 설정
└── thermal-camera/          # 열화상 코드와 선택적 YOLO 모델
```

`arda-raset`은 나머지 세 폴더를 형제 디렉터리로 찾습니다. 이 네 폴더의 상대 위치를 유지하세요. 기존 프로젝트 파일은 수정하거나 이동하지 않고 필요한 파일을 **전체 복사**했습니다.

## 3. 최초 설치

### STEP 1 — 시스템 패키지

**실행 — Jetson이 인터넷에 연결된 상태에서:**

```bash
sudo apt update
sudo apt install -y hostapd dnsmasq iw iproute2 network-manager \
  python3 python3-venv python3-pip i2c-tools curl
```

**정상 결과:** 명령이 오류 없이 끝납니다. 별도로 설치한 `uv --version`도 출력되어야 합니다.

**문제 시 확인:** `apt` 오류면 현재 Wi-Fi 또는 유선 인터넷 연결을 확인합니다. `uv`가 없다면 uv를 설치한 뒤 다음 단계로 진행합니다.

### STEP 2 — Python 의존성

**실행:**

```bash
cd ~/ARDA-2026/jetson/arda-raset
uv sync --frozen
# YOLO 사용 시에만: uv sync --frozen --extra yolo
```

**정상 결과:** `arda-raset/.venv/bin/python`이 생성됩니다. Python 3.10을 사용합니다.

**문제 시 확인:** `pyproject.toml`과 `uv.lock`의 오류 메시지를 확인합니다. YOLO용 torch/CUDA wheel은 JetPack 버전과 호환되어야 합니다. 기본 threshold 판정에는 YOLO 설치가 필요하지 않습니다.

### STEP 3 — 하드웨어와 설정

**실행:**

```bash
cd ~/ARDA-2026/jetson
nmcli device status
ls -l /dev/ttyUSB* /dev/i2c-* 2>/dev/null
id
i2cdetect -l
```

**정상 결과:** Wi-Fi 장치가 보이고, 연결된 레이더는 CLI/data USB 포트 두 개로 나타납니다. MLX90640을 사용할 I²C 버스도 보여야 합니다.

**확인할 설정:**

- `config/runtime.env`: 레이더 포트, 설치 위도·경도·방위각, 나중에 확인할 Windows 주소
- `arda-servo/config/settings.yaml`: BOARD 핀 33, 중심각, 카메라 설치 높이·기울기
- Wi-Fi 인터페이스가 `wlP1p1s0`이 아니면 두 AP 스크립트와 `hostapd.conf`, `dnsmasq.conf`의 장치명을 함께 변경
- `hostapd.conf`: SSID와 암호. GitHub 공개 전 기본 암호 변경

**문제 시 확인:** USB/I²C 장치와 일반 사용자의 접근 권한을 확인합니다. 장치가 없으면 통합기는 해당 센서를 생략할 수 있으므로 실제 실행 여부는 로그로 다시 확인해야 합니다.

### STEP 4 — 부팅 서비스 등록

**실행 — 일반 사용자로:**

```bash
cd ~/ARDA-2026/jetson
./scripts/install_service.sh
sudo systemctl start arda-jetson.service
systemctl status arda-jetson.service --no-pager
```

**정상 결과:** 서비스가 `active (running)`입니다. 다음 부팅부터 센서가 자동 시작됩니다. AP는 자동 시작하지 않습니다.

**문제 시 확인:** `journalctl -u arda-jetson.service -b -n 80`으로 Python 의존성, 장치 권한, 센서 초기화 오류를 확인합니다. 설치 스크립트는 이 폴더 안에 서비스 파일을 생성하고 systemd에 링크합니다. `/etc/systemd/system`에는 서비스 링크만 생기며 네트워크 설정 파일은 수정하지 않습니다. 폴더를 옮겼다면 이전 서비스 링크를 해제하고 새 위치에서 다시 등록해야 합니다.

## 4. 시연: Windows와 연결하기

아래 IP `192.168.10.49`는 **예시**입니다. 매 시연에서 Windows가 받은 실제 주소를 사용하세요. AP 대신 기존 Wi-Fi를 함께 사용할 경우에도 두 장치가 같은 네트워크에 있어야 하며, Windows가 그 네트워크에서 받은 IPv4 주소를 사용합니다.

### STEP 1 — Jetson AP 시작

**실행 — Jetson 로컬 터미널:**

```bash
cd ~/ARDA-2026/jetson
./scripts/start_demo.sh
nmcli device status
ip -4 addr show dev wlP1p1s0
iw dev wlP1p1s0 info
sudo tail -30 /run/arda-demo/hostapd.log
```

**정상 결과:** `ARDA_NET 준비 완료`가 출력됩니다. Wi-Fi 장치는 `unmanaged`, 주소는 `192.168.10.1/24`, `iw`는 `type AP`, hostapd 로그는 `AP-ENABLED`를 보여줍니다. 센서 서비스는 계속 `active`입니다.

**문제 시 확인:** 시작 출력의 `시작 실패 (줄 …)`, hostapd 로그, `journalctl -u arda-jetson.service -b -n 80`을 확인합니다. 
종료하려면 `sudo ./scripts/demo_stop.sh`를 실행합니다.

### STEP 2 — Windows에서 AP 접속과 IP 확인

**실행 — Windows:**

1. Wi-Fi 목록에서 `ARDA_NET`에 연결합니다.
2. PowerShell에서 다음을 실행합니다.

```powershell
ipconfig /all
ping 192.168.10.1
```

**정상 결과:** `ARDA_NET`에 연결된 **Wi-Fi 어댑터**의 IPv4가 `192.168.10.10`~`192.168.10.100`이고, DHCP 서버는 `192.168.10.1`입니다. `ping`에 응답이 옵니다. **이 IPv4 주소를 기록하세요.** 예: `192.168.10.49`.

**문제 시 확인:** `169.254.x.x`면 DHCP 실패입니다. Jetson에서 아래 로그를 확인합니다.

```bash
sudo tail -50 /run/arda-demo/hostapd.log
sudo tail -50 /run/arda-demo/dnsmasq.log
```

hostapd에는 `AP-STA-CONNECTED`와 `EAPOL-4WAY-HS-COMPLETED`, dnsmasq에는 `DHCPDISCOVER → DHCPOFFER → DHCPREQUEST → DHCPACK`가 있어야 합니다. `PSK-MISMATCH`면 Windows의 저장된 `ARDA_NET` 프로필을 지우고 암호를 다시 입력하세요.

### STEP 3 — Windows 웹 서버 시작

**실행 — Windows 네이티브 Python:**

```powershell
cd arda-algo_general
# drone_control 브랜치인지 확인
python hanriver.py
```

Windows 브라우저에서 `http://localhost:8000`을 엽니다. 관리자 PowerShell에서 TCP 8000 인바운드를 허용합니다.

```powershell
netsh advfirewall firewall add rule name="ARDA hanriver" dir=in action=allow protocol=TCP localport=8000
```

**정상 결과:** 콘솔에 `[SERVER] http://localhost:8000`이 표시되고 웹 UI가 열립니다.

**문제 시 확인:** 서버는 기본 WSL2 NAT가 아닌 Windows에서 실행하세요. Jetson에서 접근이 안 되면 Windows 방화벽과 서버의 수신 주소를 확인합니다.

### STEP 4 — Windows 주소를 Jetson에 등록

**실행 — Jetson:** STEP 2에서 기록한 주소가 `192.168.10.49`라면 설정 파일을 열 필요 없이 다음처럼 입력합니다.

```bash
cd ~/ARDA-2026/jetson
./scripts/set_report_url.sh http://192.168.10.49:8000/report
curl -m 3 http://192.168.10.49:8000/state
journalctl -u arda-jetson.service -b -n 40 --no-pager
```

`set_report_url.sh`는 센서 서비스를 재시작하며 주소를 `/run/arda-jetson/report-url`에 임시 저장합니다. `stop_demo.sh` 또는 재부팅 후에는 지워집니다. **센서를 서비스 없이 수동으로 실행한다면** 다음처럼 `--report-url` 인자를 직접 전달합니다.

```bash
./scripts/run_sensor.sh --report-url http://192.168.10.49:8000/report
```

부팅 후에도 같은 주소를 계속 쓰고 싶을 때만 `config/runtime.env`의 `ARDA_REPORT_URL`을 수정하세요. Windows IP는 DHCP로 바뀔 수 있으므로 시연마다 `ipconfig`로 다시 확인하는 편이 안전합니다.

**정상 결과:** `curl`이 HTTP 응답을 받고 센서 로그에 `웹 리포트 전송 활성화`와 실제 Windows IP가 표시됩니다. 웹 UI의 **열화상 카드**가 갱신됩니다. 낙하 확정 시 **시뮬레이션 카드**가 움직이고 로그에 `[열화상 매칭시도 N]`과 최종 판정이 표시됩니다.

**문제 시 확인:** `Connection refused`면 `hanriver.py` 실행 상태를, timeout이면 Windows IPv4·방화벽 TCP 8000·AP 연결을 확인합니다. HTTP는 되는데 이미지가 없으면 `ARDA_REPORT_URL`과 MLX90640 초기화 로그를 확인합니다. 센서를 수동 실행 중이면 `--report-url` 인자를 포함해 다시 시작하세요.

### STEP 5 — 종료와 원래 Wi-Fi 복귀

**실행 — Jetson 로컬 터미널:**

```bash
./scripts/stop_demo.sh
nmcli device status
nmcli -g GENERAL.CONNECTION,GENERAL.CON-UUID device show wlP1p1s0
ip -4 addr show dev wlP1p1s0
systemctl status arda-jetson.service --no-pager
```

**정상 결과:** `AP 종료 및 NetworkManager 복귀 완료`가 표시됩니다. Wi-Fi 장치가 이전 SSID·UUID에 다시 `connected`, `192.168.10.1`은 제거, 센서 서비스는 `inactive`입니다.

**문제 시 확인:** `sudo ./scripts/demo_stop.sh`를 다시 실행합니다. 원래 Wi-Fi가 범위 밖이거나 인증이 만료됐다면 수동으로 재연결해야 합니다. 저장된 UUID는 `/run/arda-demo/wifi.uuid`, NetworkManager 로그는 `journalctl -u NetworkManager -b`에서 확인합니다.

## 5. 수동 실행과 로그

| 목적 | 명령 |
| --- | --- |
| 센서만 실행 | `./scripts/run_sensor.sh` |
| 하드웨어 없이 확인 | `./scripts/run_sensor.sh --simulate-servo --simulate-thermal --no-radar` |
| 센서 서비스 로그 | `journalctl -u arda-jetson.service -b -f` |
| 센서 파일 로그 | `tail -f arda-raset/.logs/arda-raset.log` |
| AP 상태 | `nmcli device status` · `iw dev wlP1p1s0 info` |
| hostapd 로그 | `sudo tail -f /run/arda-demo/hostapd.log` |
| DHCP 로그 | `sudo tail -f /run/arda-demo/dnsmasq.log` |
| 서비스 상태 | `systemctl status arda-jetson.service --no-pager` |

센서 코드가 시작돼도 하드웨어가 없으면 해당 스레드는 생략될 수 있습니다. 센서 로그에서 레이더·열화상·서보 각각의 실제 시작 여부를 확인하세요. YOLO는 Jetson CUDA와 torch wheel 조합을 별도로 검증해야 합니다.
