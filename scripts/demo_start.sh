#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IFACE=wlP1p1s0
STATE=/run/arda-demo
if (( EUID != 0 )); then exec sudo -- "$0" "$@"; fi
for cmd in nmcli ip iw hostapd dnsmasq flock; do command -v "$cmd" >/dev/null || { echo "필요한 명령이 없습니다: $cmd" >&2; exit 1; }; done
[[ -d "/sys/class/net/$IFACE" ]] || { echo "무선 장치가 없습니다: $IFACE" >&2; exit 1; }
for file in "$ROOT/config/hostapd.conf" "$ROOT/config/dnsmasq.conf"; do
    grep -qx "interface=$IFACE" "$file" || { echo "interface 설정 오류: $file" >&2; exit 1; }
done
dnsmasq --test --conf-file="$ROOT/config/dnsmasq.conf"
mkdir -p -m 700 "$STATE"
exec 9>"$STATE/lock"
flock -n 9 || { echo "다른 전환 작업이 실행 중입니다" >&2; exit 1; }
[[ ! -e "$STATE/active" && ! -e "$STATE/transition" ]] || { echo "이미 AP 실행 중이거나 이전 전환이 중단되었습니다. demo_stop.sh를 실행하세요." >&2; exit 1; }
if pgrep -x hostapd >/dev/null; then echo "기존 hostapd가 실행 중입니다. 충돌 여부를 먼저 확인하세요." >&2; exit 1; fi
UUID="$(nmcli -g GENERAL.CON-UUID device show "$IFACE" | head -n 1)"
SSID="$(nmcli -g GENERAL.CONNECTION device show "$IFACE" | head -n 1)"
if [[ "$UUID" == -- ]]; then UUID=; fi
if [[ "$SSID" == -- ]]; then SSID=; fi
printf '%s\n' "$UUID" > "$STATE/wifi.uuid"
printf '%s\n' "$SSID" > "$STATE/wifi.ssid"
touch "$STATE/transition"
rollback() {
    local code=$? line=${BASH_LINENO[0]}
    trap - ERR INT TERM
    echo "시작 실패 (줄 $line, 종료 코드 $code): AP 자원을 정리하고 Wi-Fi 복귀를 시도합니다" >&2
    if [[ -f "$STATE/hostapd.log" ]]; then tail -15 "$STATE/hostapd.log" >&2; fi
    if [[ -f "$STATE/dnsmasq.log" ]]; then tail -15 "$STATE/dnsmasq.log" >&2; fi
    if [[ -f "$STATE/dnsmasq.pid" ]]; then kill "$(cat "$STATE/dnsmasq.pid")" 2>/dev/null || true; fi
    if [[ -f "$STATE/hostapd.pid" ]]; then kill "$(cat "$STATE/hostapd.pid")" 2>/dev/null || true; fi
    ip addr del 192.168.10.1/24 dev "$IFACE" 2>/dev/null || true
    nmcli device set "$IFACE" managed yes || true
    nmcli device set "$IFACE" autoconnect yes || true
    if [[ -n "$UUID" ]]; then nmcli --wait 30 connection up uuid "$UUID" ifname "$IFACE" || true; fi
    rm -f "$STATE/active" "$STATE/transition"
}
trap rollback ERR INT TERM
echo "이전 Wi-Fi: ${SSID:-없음} (UUID: ${UUID:-없음})"
echo "지금부터 무선 SSH가 끊길 수 있습니다. Jetson 로컬 터미널에서 계속 확인하세요."
nmcli device set "$IFACE" managed no
for _ in 1 2 3 4 5; do
    if nmcli -g GENERAL.STATE device show "$IFACE" | grep -q unmanaged; then break; fi
    sleep 1
done
if ! nmcli -g GENERAL.STATE device show "$IFACE" | grep -q unmanaged; then
    echo "NetworkManager가 장치를 놓지 않았습니다" >&2; false
fi
ip link set "$IFACE" down
ip addr flush dev "$IFACE"
sleep 2
ip link set "$IFACE" up
ip addr add 192.168.10.1/24 dev "$IFACE"
: > "$STATE/hostapd.log"
: > "$STATE/dnsmasq.log"
hostapd -B -P "$STATE/hostapd.pid" -f "$STATE/hostapd.log" "$ROOT/config/hostapd.conf" 9>&-
sleep 2
kill -0 "$(cat "$STATE/hostapd.pid")"
iw dev "$IFACE" info | grep -q 'type AP'
grep -q 'AP-ENABLED' "$STATE/hostapd.log"
dnsmasq --conf-file="$ROOT/config/dnsmasq.conf" --no-daemon 9>&- >"$STATE/dnsmasq.log" 2>&1 &
printf '%s\n' "$!" > "$STATE/dnsmasq.pid"
sleep 2
kill -0 "$(cat "$STATE/dnsmasq.pid")"
touch "$STATE/active"
rm -f "$STATE/transition"
trap - ERR INT TERM
echo "ARDA_NET 준비 완료: 192.168.10.1/24"
echo "로그: $STATE/hostapd.log, $STATE/dnsmasq.log"
