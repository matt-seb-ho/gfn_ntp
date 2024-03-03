import json
import os

REPO_ROOT = "/mnt/hdd/msho/gfn_ntp/"

def prepend_repo_root(p: str) -> str:
    return os.path.join(REPO_ROOT, p)

def _pp(d):
    print(json.dumps(d, indent=2))