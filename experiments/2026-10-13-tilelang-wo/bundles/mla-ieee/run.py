"""Route equality and time per route, with the sparse MLA's fast math on and off."""
import os
import subprocess
import sys

codes = {}
for fast in ("1", "0"):
    env = dict(os.environ, VLLM_DS41_MLA_FAST_MATH=fast)
    codes[f"routes fast={fast}"] = subprocess.call([sys.executable, "/b/check_mla_routes.py"], env=env)
    codes[f"time fast={fast}"] = subprocess.call([sys.executable, "/b/time_mla.py"], env=env)
print("exit codes", codes, flush=True)
# The IEEE build must keep every route identical; the fast-math one is expected to differ.
sys.exit(1 if codes["routes fast=0"] or codes["time fast=0"] or codes["time fast=1"] else 0)
