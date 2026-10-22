#!/usr/bin/env bash
# Receipts for experiments/2026-10-20-dsv41-control-spellings, run on one Spark with the
# r6x image, no GPU and no network. Writes to ./runs.
set -u
IMG=vllm-ds41f-kkref:19f2c20ed4d6-r6x
SRC=/opt/spark3/candidate/vllm
B=$PWD/bundles/tokenizer-tests
CAND="-v $B/vllm_tokenizers_deepseek_v41.py:$SRC/vllm/tokenizers/deepseek_v41.py:ro -v $B/vllm_renderers_deepseek_v41.py:$SRC/vllm/renderers/deepseek_v41.py:ro"
TESTS="-v $B/tests_tokenizers__test_deepseek_v41.py:$SRC/tests/tokenizers_/test_deepseek_v41.py:ro"
FILES="tests/parser/engine/test_deepseek_v4.py tests/parser/engine/test_deepseek_v41.py tests/tokenizers_/test_deepseek_v4.py tests/tokenizers_/test_deepseek_v41.py tests/tokenizers_/test_deepseek_v41_effort.py"
PYTEST="python3 -m pytest -q -rfE -p no:cacheprovider --noconftest -p tests.parser.engine.conftest"
mkdir -p runs
run() {  # name, mounts
  local name=$1; shift
  {
    echo "\$ image $IMG $(docker image inspect -f '{{.Id}}' $IMG)"
    echo "\$ mounts: ${*:-none}"
    echo "\$ $PYTEST $FILES"
    docker run --rm --memory=3g --network none --entrypoint bash "$@" $IMG \
      -c "cd $SRC && timeout 600 $PYTEST $FILES 2>&1 | sed 's/ - .*//' | grep -E '^(FAILED|ERROR)|passed|failed'"
  } > runs/$name.txt 2>&1
  tail -1 runs/$name.txt
}
run tests-r6x                               # the image as promoted
run tests-r6x-with-new-tests $TESTS         # new tests against the promoted tokenizer
run tests-candidate $CAND $TESTS            # candidate tokenizer and tests
{
  echo "\$ python3 overhead.py <promoted deepseek_v41.py> <candidate deepseek_v41.py> 21"
  docker run --rm --memory=4g --network none --entrypoint python3 -v $PWD:/w:ro \
    -v $HOME/models/DeepSeek-V4.1-Flash:/models:ro -w / $IMG \
    /w/overhead.py $SRC/vllm/tokenizers/deepseek_v41.py /w/bundles/tokenizer-tests/vllm_tokenizers_deepseek_v41.py 21 2>&1 | grep -E "tok "
} > runs/overhead.txt
cat runs/overhead.txt
