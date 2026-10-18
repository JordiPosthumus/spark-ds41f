#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER_NAME:?CONTAINER_NAME is required}"
: "${THRESHOLD_GIB:?THRESHOLD_GIB is required}"
: "${INTERVAL:?INTERVAL is required}"
: "${MEMGUARD_PHASE:=steady}"
: "${WAIT_FOR_CONTAINER_SECONDS:=0}"
: "${MEMGUARD_SUSTAINED_SAMPLES:=1}"
: "${MEMGUARD_WARN_GIB:=0}"
: "${READY_FILE:=}"

if ! [[ "$THRESHOLD_GIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "THRESHOLD_GIB must be a positive integer" >&2
  exit 2
fi
if ! [[ "$INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$INTERVAL" =~ ^0+([.]0+)?$ ]]; then
  echo "INTERVAL must be a positive number" >&2
  exit 2
fi
if ! [[ "$WAIT_FOR_CONTAINER_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "WAIT_FOR_CONTAINER_SECONDS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$MEMGUARD_SUSTAINED_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MEMGUARD_SUSTAINED_SAMPLES must be a positive integer" >&2
  exit 2
fi
if ! [[ "$MEMGUARD_WARN_GIB" =~ ^[0-9]+$ ]]; then
  echo "MEMGUARD_WARN_GIB must be a non-negative integer" >&2
  exit 2
fi
if ((MEMGUARD_WARN_GIB > 0 && MEMGUARD_WARN_GIB <= THRESHOLD_GIB)); then
  echo "MEMGUARD_WARN_GIB must exceed THRESHOLD_GIB" >&2
  exit 2
fi

threshold_kib=$((THRESHOLD_GIB * 1024 * 1024))
warn_kib=$((MEMGUARD_WARN_GIB * 1024 * 1024))
deadline=$((SECONDS + WAIT_FOR_CONTAINER_SECONDS))
observed_running=false
ready=false
breaches=0
warned=false

while true; do
  inspectable=true
  if ! running=$(docker inspect --format '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null); then
    inspectable=false
    running=
  fi
  if [[ "$inspectable" == "false" && "$observed_running" == "true" ]]; then
    logger -t spark-memguard \
      "cannot inspect $CONTAINER_NAME after startup during $MEMGUARD_PHASE; failing closed"
    docker kill --signal KILL "$CONTAINER_NAME" >/dev/null 2>&1 || true
    exit 1
  fi
  if [[ "$running" == "true" ]]; then
    observed_running=true
  elif [[ "$observed_running" == "true" ]]; then
    exit 0
  elif ((SECONDS >= deadline)); then
    logger -t spark-memguard \
      "$CONTAINER_NAME did not start within ${WAIT_FOR_CONTAINER_SECONDS}s; leaving no unbounded watcher"
    exit 1
  fi

  available_kib=$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)
  if [[ -z "$available_kib" ]]; then
    logger -t spark-memguard \
      "cannot read MemAvailable during $MEMGUARD_PHASE; failing closed"
    if [[ "$running" == "true" ]]; then
      docker kill --signal KILL "$CONTAINER_NAME" >/dev/null
    fi
    exit 1
  fi
  if ((available_kib < threshold_kib)); then
    if [[ "$running" != "true" ]]; then
      logger -t spark-memguard \
        "refusing to wait for $CONTAINER_NAME during $MEMGUARD_PHASE: MemAvailable ${available_kib} KiB below ${threshold_kib} KiB"
      exit 1
    fi
    breaches=$((breaches + 1))
    if ((breaches >= MEMGUARD_SUSTAINED_SAMPLES)); then
      logger -t spark-memguard \
        "killing $CONTAINER_NAME during $MEMGUARD_PHASE: MemAvailable ${available_kib} KiB below ${threshold_kib} KiB on ${breaches} consecutive samples"
      docker kill --signal KILL "$CONTAINER_NAME" >/dev/null
      exit 1
    fi
    logger -t spark-memguard \
      "pressure on $CONTAINER_NAME during $MEMGUARD_PHASE: MemAvailable ${available_kib} KiB below ${threshold_kib} KiB (sample ${breaches} of ${MEMGUARD_SUSTAINED_SAMPLES}; container kept running)"
  else
    if ((breaches > 0)); then
      logger -t spark-memguard \
        "$CONTAINER_NAME during $MEMGUARD_PHASE recovered above ${threshold_kib} KiB after ${breaches} consecutive samples"
    fi
    breaches=0
    if ((warn_kib > 0)) && ((available_kib < warn_kib)); then
      if [[ "$warned" == "false" ]]; then
        logger -t spark-memguard \
          "low headroom on $CONTAINER_NAME during $MEMGUARD_PHASE: MemAvailable ${available_kib} KiB below ${warn_kib} KiB"
        warned=true
      fi
    elif [[ "$warned" == "true" ]]; then
      warned=false
    fi
  fi

  if [[ "$ready" == "false" ]]; then
    if [[ -n "$READY_FILE" ]]; then
      : >"$READY_FILE"
    fi
    ready=true
  fi
  sleep "$INTERVAL"
done
