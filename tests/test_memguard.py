"""Host-independent behavioral tests for the fail-closed memory guard."""

from __future__ import annotations

import importlib.machinery
import json
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMGUARD = ROOT / "scripts" / "memguard.sh"
MEMGUARD_START = ROOT / "scripts" / "memguard-start.sh"
loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
spec = importlib.util.spec_from_loader("spark", loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)


class MemguardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.state = self.root / "state"
        self.actions = self.root / "actions"
        self.ready = self.root / "ready"
        self._write_command(
            "docker",
            """#!/usr/bin/env bash
set -eu
case "$1" in
  inspect)
    state=$(cat "$FAKE_STATE")
    if [[ "$state" == absent ]]; then
      exit 1
    fi
    printf '%s\n' "$state"
    ;;
  kill)
    printf 'kill %s\n' "$*" >>"$FAKE_ACTIONS"
    printf 'false\n' >"$FAKE_STATE"
    ;;
  *)
    exit 2
    ;;
esac
""",
        )
        self._write_command(
            "awk",
            """#!/usr/bin/env bash
set -eu
if [[ -n "${FAKE_MEM_SEQUENCE:-}" ]]; then
  counter="$FAKE_STATE.calls"
  calls=$(cat "$counter" 2>/dev/null || printf '0')
  printf '%s\n' "$((calls + 1))" >"$counter"
  IFS=',' read -r -a values <<<"$FAKE_MEM_SEQUENCE"
  index=$calls
  if ((index >= ${#values[@]})); then
    index=$((${#values[@]} - 1))
  fi
  printf '%s\n' "$((values[index] * 1024 * 1024))"
  exit 0
fi
printf '%s\n' "$FAKE_MEM_AVAILABLE_KIB"
""",
        )
        self._write_command(
            "logger",
            """#!/usr/bin/env bash
set -eu
printf 'logger %s\n' "$*" >>"$FAKE_ACTIONS"
""",
        )
        self._write_command(
            "sleep",
            """#!/usr/bin/env bash
set -eu
state=$(cat "$FAKE_STATE")
if [[ -n "${FAKE_SLEEP_LIMIT:-}" ]]; then
  counter="$FAKE_STATE.sleeps"
  pauses=$(cat "$counter" 2>/dev/null || printf '0')
  pauses=$((pauses + 1))
  printf '%s\n' "$pauses" >"$counter"
  if ((pauses >= FAKE_SLEEP_LIMIT)); then
    printf 'false\n' >"$FAKE_STATE"
  fi
  exit 0
fi
case "$state" in
  absent) printf 'true\n' >"$FAKE_STATE" ;;
  true) printf 'false\n' >"$FAKE_STATE" ;;
esac
""",
        )
        self._write_command(
            "sudo",
            """#!/usr/bin/env bash
set -eu
printf 'sudo %s\n' "$*" >>"$FAKE_ACTIONS"
if [[ "$*" == *" systemd-run "* ]]; then
  for argument in "$@"; do
    case "$argument" in
      --setenv=READY_FILE=*)
        ready=${argument#--setenv=READY_FILE=}
        : >"$ready"
        ;;
    esac
  done
fi
""",
        )

    def _write_command(self, name: str, body: str) -> None:
        command = self.bin / name
        command.write_text(body, encoding="utf-8")
        command.chmod(0o755)

    def _run(
        self,
        *,
        state: str,
        available_gib: int,
        wait_seconds: int,
        phase: str = "startup",
        sustained_samples: int | None = None,
        warn_gib: int | None = None,
        mem_sequence: list[int] | None = None,
        sleep_limit: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.state.write_text(f"{state}\n", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin}:{environment['PATH']}",
                "CONTAINER_NAME": "candidate",
                "THRESHOLD_GIB": "5",
                "INTERVAL": "0.25",
                "MEMGUARD_PHASE": phase,
                "WAIT_FOR_CONTAINER_SECONDS": str(wait_seconds),
                "READY_FILE": str(self.ready),
                "FAKE_STATE": str(self.state),
                "FAKE_ACTIONS": str(self.actions),
                "FAKE_MEM_AVAILABLE_KIB": str(available_gib * 1024 * 1024),
            }
        )
        if sustained_samples is not None:
            environment["MEMGUARD_SUSTAINED_SAMPLES"] = str(sustained_samples)
        if warn_gib is not None:
            environment["MEMGUARD_WARN_GIB"] = str(warn_gib)
        if mem_sequence is not None:
            environment["FAKE_MEM_SEQUENCE"] = ",".join(
                str(value) for value in mem_sequence
            )
        if sleep_limit is not None:
            environment["FAKE_SLEEP_LIMIT"] = str(sleep_limit)
        return subprocess.run(
            [str(MEMGUARD)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )

    def test_prelaunch_guard_is_ready_before_container_and_tracks_its_lifetime(
        self,
    ) -> None:
        result = self._run(state="absent", available_gib=16, wait_seconds=120)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.ready.exists())
        self.assertFalse(self.actions.exists())

    def test_running_container_is_killed_immediately_below_threshold(self) -> None:
        result = self._run(state="true", available_gib=4, wait_seconds=0)

        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.ready.exists())
        self.assertIn("kill kill --signal KILL candidate", self.actions.read_text())

    def test_prelaunch_guard_rejects_low_memory_without_starting_container(
        self,
    ) -> None:
        result = self._run(state="absent", available_gib=4, wait_seconds=120)

        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.ready.exists())
        self.assertNotIn("kill --signal", self.actions.read_text())

    def _start(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin}:{environment['PATH']}",
                "XDG_RUNTIME_DIR": str(self.root),
                "FAKE_ACTIONS": str(self.actions),
            }
        )
        return subprocess.run(
            [str(MEMGUARD_START), *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )

    def test_transient_pressure_keeps_the_container_running(self) -> None:
        result = self._run(
            state="true",
            available_gib=4,
            wait_seconds=0,
            phase="steady",
            sustained_samples=3,
            mem_sequence=[4, 16, 16],
            sleep_limit=3,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        actions = self.actions.read_text(encoding="utf-8")
        self.assertNotIn("kill --signal", actions)
        self.assertIn("pressure on candidate during steady", actions)
        self.assertIn("sample 1 of 3", actions)
        self.assertIn("recovered above", actions)

    def test_sustained_pressure_kills_on_the_configured_sample(self) -> None:
        result = self._run(
            state="true",
            available_gib=4,
            wait_seconds=0,
            phase="steady",
            sustained_samples=3,
            mem_sequence=[4, 4, 4],
            sleep_limit=5,
        )

        self.assertEqual(result.returncode, 1)
        actions = self.actions.read_text(encoding="utf-8")
        self.assertIn("kill kill --signal KILL candidate", actions)
        self.assertIn("on 3 consecutive samples", actions)
        self.assertEqual(actions.count("sample 1 of 3"), 1)
        self.assertEqual(actions.count("sample 2 of 3"), 1)

    def test_low_headroom_warns_once_per_crossing(self) -> None:
        result = self._run(
            state="true",
            available_gib=6,
            wait_seconds=0,
            phase="steady",
            sustained_samples=3,
            warn_gib=8,
            mem_sequence=[6, 6],
            sleep_limit=2,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        actions = self.actions.read_text(encoding="utf-8")
        self.assertNotIn("kill --signal", actions)
        self.assertEqual(actions.count("low headroom on candidate"), 1)

    def test_warn_threshold_above_the_kill_line_is_required(self) -> None:
        result = self._run(
            state="true",
            available_gib=16,
            wait_seconds=0,
            phase="steady",
            warn_gib=5,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("MEMGUARD_WARN_GIB must exceed THRESHOLD_GIB", result.stderr)

    def test_start_wrapper_confirms_protected_prelaunch_unit(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin}:{environment['PATH']}",
                "XDG_RUNTIME_DIR": str(self.root),
                "FAKE_ACTIONS": str(self.actions),
            }
        )
        return subprocess.run(
            [str(MEMGUARD_START), *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )

    def test_start_wrapper_confirms_protected_prelaunch_unit(self) -> None:
        # An arm overlay: the guard's settings come from the resolved configuration
        # (r6m's first boot failed to arm when the script read the arm's file).
        cluster = spark.read_cluster("experiments/2026-10-14-r6m/r6m-release.json")
        result = self._start(*spark.memguard_settings(cluster, "startup", "prelaunch"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode prelaunch", result.stdout)
        actions = self.actions.read_text(encoding="utf-8")
        self.assertIn("--property=OOMScoreAdjust=-1000", actions)
        self.assertIn("--property=MemoryMin=64M", actions)
        self.assertIn("--property=CPUWeight=1000", actions)
        self.assertIn("--setenv=INTERVAL=0.25", actions)
        self.assertIn("--setenv=WAIT_FOR_CONTAINER_SECONDS=120", actions)
        self.assertIn("--setenv=CONTAINER_NAME=dsv41-karmic-kraken", actions)
        self.assertIn("--setenv=MEMGUARD_SUSTAINED_SAMPLES=1", actions)
        self.assertIn("--setenv=MEMGUARD_WARN_GIB=0", actions)

    def test_start_wrapper_passes_the_pressure_policy(self) -> None:
        cluster = spark.read_cluster("experiments/2026-10-14-r6m/r6m-release.json")
        cluster["host"]["memguard_sustained_samples"] = 3
        cluster["host"]["memguard_warn_gib"] = 4
        path = self.root / "pressure-policy.json"
        path.write_text(json.dumps(cluster), encoding="utf-8")
        configured = spark.read_cluster(str(path))

        steady = self._start(*spark.memguard_settings(configured, "steady", "prelaunch"))
        self.assertEqual(steady.returncode, 2)  # prelaunch is startup-only; arity is what matters
        startup = self._start(*spark.memguard_settings(configured, "startup", "prelaunch"))
        self.assertEqual(startup.returncode, 0, startup.stderr)
        actions = self.actions.read_text(encoding="utf-8")
        self.assertIn("--setenv=MEMGUARD_SUSTAINED_SAMPLES=3", actions)
        # The warn line is steady-only: the startup guard runs at the launch threshold.
        self.assertIn("--setenv=MEMGUARD_WARN_GIB=0", actions)

    def test_start_wrapper_refuses_a_config_path(self) -> None:
        result = self._start("config/cluster.json", "startup", "prelaunch")

        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)
        self.assertFalse(self.actions.exists())

    def test_steady_settings_have_no_container_wait(self) -> None:
        cluster = spark.read_cluster("experiments/2026-10-14-r6m/r6m-release.json")
        name, phase, gib, interval, wait, mode, sustained, warn = spark.memguard_settings(
            cluster, "steady", "running"
        )
        self.assertEqual((name, phase, wait), ("dsv41-karmic-kraken", "steady", "0"))
        self.assertEqual(gib, str(cluster["host"]["memguard_gib"]))
        self.assertEqual(interval, str(cluster["host"]["memguard_interval_seconds"]))
        self.assertEqual((mode, sustained, warn), ("running", "1", "0"))


if __name__ == "__main__":
    unittest.main()
