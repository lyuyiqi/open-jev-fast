#!/usr/bin/env bash
# Start the accelerated server (same HTTP API as `python -m jev.server`, POST /v1/systemone).
# Required env:
#   OPEN_JEV_DIR  Open-Jev checkout (https://github.com/Zefan-Cai/Open-Jev), installed in the active Python env
#   OJ_CKPT       Open-Jev-27B-v1.1 checkpoint dir (.../package/checkpoint of ZefanCai/Open-Jev-27B-v1.1)
#   CUDA_HOME     CUDA 13 root with bin/nvcc, include/ and lib/libcublasLt.so (pip: site-packages/nvidia/cu13)
# Optional: PORT (18791), ACCESS_LOG (access.log), MAX_GRAPHS (512), OJ_USE_LT (1 = cuBLASLt autotune)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OPEN_JEV_DIR:?set OPEN_JEV_DIR}" "${OJ_CKPT:?set OJ_CKPT}" "${CUDA_HOME:?set CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH" OJ_USE_LT="${OJ_USE_LT:-1}" MAX_GRAPHS="${MAX_GRAPHS:-512}"
cd "$OPEN_JEV_DIR"
exec python -u "$HERE/src/server.py" --checkpoint "$OJ_CKPT" --device cuda:0 --max-length 16384 \
  --host 0.0.0.0 --port "${PORT:-18791}"
