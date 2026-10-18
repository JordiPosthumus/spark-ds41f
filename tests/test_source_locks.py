"""An experiment's upstream lock and its source manifest pin the same revisions: the
build checks out the lock's revision and checks the manifest's tree, so a lock left at
an older revision fails the image build (r6n, 2026-10-15) instead of failing here."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceLockTest(unittest.TestCase):
    def test_locks_and_manifests_pin_the_same_revisions(self) -> None:
        checked, problems = 0, []
        for lock_path in sorted(ROOT.glob("experiments/*/upstreams.lock.json")):
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            manifest_path = ROOT / lock.get("source_manifest", "")
            if not lock.get("source_manifest") or not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, source in lock.get("sources", {}).items():
                pinned = manifest.get(name, {})
                base = pinned.get("base_revision") if isinstance(pinned, dict) else None
                if not source.get("revision") or not base:
                    continue
                checked += 1
                if source["revision"] != base:
                    problems.append(f"{lock_path.relative_to(ROOT)}: {name} {source['revision'][:12]} "
                                    f"but {manifest_path.relative_to(ROOT)} {base[:12]}")
        self.assertGreater(checked, 0)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
