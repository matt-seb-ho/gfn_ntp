# TODO: fix this to not depend on utils

import os
import sys

# Assuming the script is running from the scripts directory and src is at the root
project_src_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, project_src_path)

from huggingface_hub import login
from utils import get_hf_access_token

def main():
    HF_ACCESS_TOKEN = get_hf_access_token()
    # login(token=HF_ACCESS_TOKEN, add_to_git_credential=True)
    # nlp server result:
    """
    Token is valid (permission: write).
    Cannot authenticate through git-credential as no helper is defined on your machine.
    You might have to re-authenticate when pushing to the Hugging Face Hub.
    Run the following command in your terminal in case you want to set the 'store' credential helper as default.

    git config --global credential.helper store

    Read https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more details.
    Token has not been saved to git credential helper.
    Your token has been saved to /mnt/hdd/msho/.cache/huggingface/token
    Login successful
    """

    login(token=HF_ACCESS_TOKEN)
    print("finished logging in I guess...")

if __name__ == "__main__":
    main()
