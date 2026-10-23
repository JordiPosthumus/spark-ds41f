#!/usr/bin/env bash
# Runs the CPU reference twice, one after the other: WO-A input quantization off, then on.
set -u
cd ~/spark-lab/json-count
SNAP=/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
for v in 0 1; do
  name=json-count-ref-woa$v-v2
  docker run --name $name --memory 6g --memory-swap 6g --cpus 10 --cpuset-cpus 0-4,10-14 --network none \
    --device-read-bps /dev/nvme0n1:800mb -e HF_HUB_OFFLINE=1 -e REF_THREADS=10 -e REF_WO_A_QUANT=$v \
    -v $HOME/.cache/huggingface/hub:/hub:ro -v $PWD:/work:ro -v $PWD/out:/out \
    --entrypoint python3 vllm-ds41f-kkref:19f2c20ed4d6-r6v -I /work/ref_logits.py $SNAP /out/ref-woa$v.json \
    > out/ref-woa$v.log 2>&1
  echo "$name exit $?" >> out/pair.status
  docker rm $name > /dev/null
done
echo DONE >> out/pair.status
