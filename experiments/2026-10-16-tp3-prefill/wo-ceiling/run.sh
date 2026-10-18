#!/usr/bin/env bash
# Run the WO ceiling bench on this node (dgx4, off the fabric) in the r6n image, if the GPU is free.
set -euo pipefail
B=$HOME/spark-lab/wo-ceiling
for f in ~/spark-request.json ~/spark3-request.json; do [ -e "$f" ] && { echo "refusing: $f exists"; exit 1; }; done
[ -z "$(docker ps -q)" ] || { echo "refusing: a container is running"; docker ps; exit 1; }
mkdir -p "$B/cache" "$B/results"
out="$B/results/$(date -u +%Y%m%dT%H%M%SZ).log"
timeout 3h docker run --rm --name wo-ceiling --gpus all --ipc=host --memory=48g --memory-swap=48g \
  -e HF_HUB_OFFLINE=1 -e CUTE_DSL_ARCH=sm_121a -e TILELANG_CACHE_DIR=/kcache/tilelang \
  -e CUTE_DSL_CACHE_DIR=/kcache/cute -e B12X_COMPILE_CACHE_DIR=/kcache/b12x -e B12X_CUTE_COMPILE_CACHE_DIR=/kcache/b12x-cute \
  -e TRITON_CACHE_DIR=/kcache/triton \
  -v "$B/bench_wo_ceiling.py:/b/bench_wo_ceiling.py:ro" -v "$B/kbench.py:/b/kbench.py:ro" -v "$B/cache:/kcache" \
  -w /opt/spark3/candidate/vllm --entrypoint python3 vllm-ds41f-kkref:19f2c20ed4d6-r6n /b/bench_wo_ceiling.py \
  2>&1 | tee "$out"
echo "log: $out"
