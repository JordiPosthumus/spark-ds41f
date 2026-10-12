"""Read-only local file verification for a future isolated Ring 2 canary.

This never imports vLLM, launches anything or sends a request.
"""
import argparse
import hashlib
import json
from pathlib import Path

from transform import restore

SERVING = 'entrypoints/openai/chat_completion/serving.py'
PACKAGE = 'entrypoints/openai/chat_completion/ds41f_mechanism_wiring'


def inspect(root, manifest, mounted=False):
    checks = []
    for relative, expected in manifest['native_source_sha256'].items():
        path = Path(root) / relative
        try:
            if relative == SERVING and mounted:
                data = restore(path.read_text(), manifest['plan_sha256']).encode()
            else:
                data = path.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
        except (OSError, ValueError):
            actual = None
        checks.append({'file': relative, 'expected_sha256': expected, 'actual_sha256': actual, 'match': actual == expected})
    if mounted:
        prefix = 'overlay/vllm/' + PACKAGE + '/'
        for relative, expected in manifest['files'].items():
            if relative.startswith(prefix):
                path = Path(root) / relative.removeprefix('overlay/vllm/')
                actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                checks.append({'file': str(path.relative_to(root)), 'expected_sha256': expected,
                               'actual_sha256': actual, 'match': actual == expected})
    return {'matches': all(c['match'] for c in checks), 'mounted': mounted, 'checks': checks,
            'runtime_objects_verified': False, 'container_config_verified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--mounted', action='store_true')
    args = parser.parse_args()
    result = inspect(args.installed_root, json.loads(args.manifest.read_text()), args.mounted)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['matches'] else 1)
