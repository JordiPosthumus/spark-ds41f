"""Our deployments must never cap images per prompt.

The per-prompt image cap (``--limit-mm-per-prompt``) ships in every upstream recipe and
has reappeared in our site overlays more than once, breaking multi-image requests with
``400: At most 4 image(s) may be provided in one prompt``.

A site overlay (``config/*.local.json``) replaces the base ``serve_args`` list wholesale
when the command is rendered, so the cap can only reach a launch through a site overlay.
This test makes that regression impossible to land unnoticed: any site config that sets a
per-prompt image cap fails the suite. It is deliberately about *our* overlays, not the
upstream recipes, which legitimately ship the capped default.
"""

import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLAG = "--limit-mm-per-prompt"


def capped_configs(paths):
    """Site configs whose serve_args set a per-prompt image cap.

    Returns ``{path_as_str: cap_value}`` for every config that carries the flag.
    Unreadable or malformed files are ignored here so the caller reports on the flag,
    not on unrelated file problems.
    """
    offenders = {}
    for path in paths:
        try:
            doc = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        args = doc.get("serve_args")
        if isinstance(args, list) and FLAG in args:
            index = args.index(FLAG)
            offenders[str(path)] = args[index + 1] if index + 1 < len(args) else None
    return offenders


class SiteImageCapTest(unittest.TestCase):
    def test_no_site_config_caps_images_per_prompt(self):
        sites = sorted(ROOT.glob("config/*.local.json"))
        if not sites:
            # Site overlays are gitignored and absent in a fresh clone; the checker's
            # own coverage lives in test_checker_detects_a_cap below.
            self.skipTest("no site config present (config/*.local.json)")
        self.assertEqual(
            capped_configs(sites), {},
            "site config caps images per prompt; drop --limit-mm-per-prompt so "
            "multi-image requests keep working (the cap has caused 400s before)",
        )

    def test_checker_detects_a_cap(self):
        # A checker that cannot fail is worthless: prove it flags a capped overlay and
        # clears a clean one. This is what keeps the skip above from hiding a regression.
        with tempfile.TemporaryDirectory() as temporary:
            capped = Path(temporary) / "site-capped.local.json"
            capped.write_text(json.dumps({"serve_args": ["/models", FLAG, '{"image":4}']}))
            self.assertEqual(capped_configs([capped]), {str(capped): '{"image":4}'})

            clean = Path(temporary) / "site-clean.local.json"
            clean.write_text(json.dumps({"serve_args": ["/models", "--dtype", "bfloat16"]}))
            self.assertEqual(capped_configs([clean]), {})

            broken = Path(temporary) / "site-broken.local.json"
            broken.write_text("{not json")
            self.assertEqual(capped_configs([broken]), {})


if __name__ == "__main__":
    unittest.main()
