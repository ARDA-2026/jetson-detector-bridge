#!/usr/bin/env bash
# jetson-ai-lab 인덱스의 torch(requirements-yolo-jetson.txt)가 dlopen하는
# libcudss.so.0을 자체적으로 포함하지 않아서 생기는 Jetson 전용 문제를
# 보완함 — nvidia-cudss-cu12 wheel의 .so 파일을 torch/lib에 직접 복사한다.
# 라즈베리파이는 이 스크립트가 필요 없다.
set -euo pipefail
cd "$(dirname "$0")"

CUDSS_VERSION="0.8.0.10"
TORCH_LIB="mlx_env_jetson/lib/python3.10/site-packages/torch/lib"

if [ ! -d "$TORCH_LIB" ]; then
    echo "error: $TORCH_LIB not found — 'mlx_env_jetson/bin/pip install -r requirements-yolo-jetson.txt' 먼저 실행하세요" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -sL -o "$tmp/cudss.whl" \
    "https://files.pythonhosted.org/packages/py3/n/nvidia-cudss-cu12/nvidia_cudss_cu12-${CUDSS_VERSION}-py3-none-manylinux_2_17_aarch64.whl"
unzip -q "$tmp/cudss.whl" -d "$tmp/extracted"
cp -f "$tmp"/extracted/nvidia/cu12/lib/libcudss*.so.0 "$TORCH_LIB/"
echo "libcudss.so.0 을 $TORCH_LIB 에 설치했습니다."
