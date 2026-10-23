#!/usr/bin/env bash
# After the pair finishes: the reference as shipped with FP64 accumulation (the noise-floor control).
set -u
until grep -q DONE ~/spark-lab/json-count/out/pair.status 2>/dev/null; do sleep 15; done
cd ~/spark-lab/json-count-fp64
SNAP=/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
OUTDIR=$HOME/spark-lab/json-count/out
docker run --name json-count-ref-fp64 --memory 6g --memory-swap 6g --cpus 10 --cpuset-cpus 0-4,10-14 --network none \
  --device-read-bps /dev/nvme0n1:800mb -e HF_HUB_OFFLINE=1 -e REF_THREADS=10 -e REF_WO_A_QUANT=1 -e REF_ACCUM=float64 \
  -v $HOME/.cache/huggingface/hub:/hub:ro -v $PWD:/work:ro -v $OUTDIR:/out \
  --entrypoint python3 vllm-ds41f-kkref:19f2c20ed4d6-r6v -I /work/ref_logits.py $SNAP /out/ref-woa1-fp64.json \
  > $OUTDIR/ref-woa1-fp64.log 2>&1
echo "json-count-ref-fp64 exit $?" >> $OUTDIR/pair.status
docker rm json-count-ref-fp64 > /dev/null
echo FP64DONE >> $OUTDIR/pair.status
