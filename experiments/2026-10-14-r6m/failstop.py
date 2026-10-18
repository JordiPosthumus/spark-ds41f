"""Fabric script: sparknet's fail-stop scenario on this rank.

The lab runs it on every node at once in the arm's image (``fabric`` job) with
the probe's rank arguments; ``FAILSTOP_SCENARIO`` (``eager`` or ``graph``)
picks the scenario. ``sparknet.probe.failstop`` prints the rank's report and
exits 0 when the rank stopped its GPU within the timeout and named the stalled
peer.
"""

import argparse
import os
import sys

from sparknet.probe import failstop


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for option in ("--rank", "--world-size", "--master-addr", "--master-port"):
        parser.add_argument(option, required=True)
    args, _ = parser.parse_known_args(argv)
    return failstop.main([os.environ.get("FAILSTOP_SCENARIO", "eager"), "--rank", args.rank,
                          "--world-size", args.world_size, "--master-addr", args.master_addr,
                          "--master-port", args.master_port])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
