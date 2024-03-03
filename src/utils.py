import json
import os
from dotenv import load_dotenv

REPO_ROOT = "/mnt/hdd/msho/gfn_ntp/"
HF_ACCESS_TOKEN_VAR_NAME = "HF_ACCESS_TOKEN"

def prepend_repo_root(p: str) -> str:
    return os.path.join(REPO_ROOT, p)

def _pp(d):
    print(json.dumps(d, indent=2))

def get_hf_access_token():
    load_dotenv()
    token = os.getenv(HF_ACCESS_TOKEN_VAR_NAME)
    return token