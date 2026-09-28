#!/usr/bin/env bash
set -Eeuo pipefail
IFACE=wlP1p1s0
STATE=/run/arda-demo
if (( EUID != 0 )); then exec sudo -- "$0" "$@"; fi
mkdir -p -m 700 "$STATE"
exec 9>"$STATE/lock"
flock -n 9 || { echo "다른 전환 작업이 실행 중입니다" >&2; exit 1; }
stop_pid() {
    local file=$1 expected=$2 pid
    [[ -f "$file" ]] || return 0
    pid="$(cat "$file")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$pid/comm" ]] && [[ "$(cat "/proc/$pid/comm")" == "$expected" ]]; then
        kill "$pid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    fi
    rm -f "$file"
}
stop_pid "$STATE/dnsmasq.pid" dnsmasq
stop_pid "$STATE/hostapd.pid" hostapd
if [[ -d "/sys/class/net/$IFACE" ]]; then
    ip addr del 192.168.10.1/24 dev "$IFACE" 2>/dev/null || true
    nmcli device set "$IFACE" managed yes
    nmcli device set "$IFACE" autoconnect yes
fi
UUID="$(cat "$STATE/wifi.uuid" 2>/dev/null || true)"
if [[ -n "$UUID" && -d "/sys/class/net/$IFACE" ]]; then
    echo "이전 Wi-Fi 프로필 복귀: $UUID"
    if ! nmcli --wait 30 connection up uuid "$UUID" ifname "$IFACE"; then
        echo "복귀 실패. 명령을 다시 실행하거나 저장된 UUID로 nmcli connection up uuid '$UUID' ifname '$IFACE'를 실행하세요." >&2
        exit 1
    fi
else
    echo "이전 활성 Wi-Fi가 없어 NetworkManager 자동 연결을 사용합니다."
fi
rm -f "$STATE/active" "$STATE/transition" "$STATE/wifi.uuid" "$STATE/wifi.ssid"
echo "AP 종료 및 NetworkManager 복귀 완료"
