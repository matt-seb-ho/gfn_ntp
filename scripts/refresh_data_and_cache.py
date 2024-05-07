# refreshes lean_dojo data and cache.
import argparse
import json
import os
import shutil
from hashlib import md5
from pathlib import Path

from loguru import logger

from load_gh_token import load_github_access_token

load_github_access_token()
from lean_dojo import LeanGitRepo, is_available_in_cache # isort: skip

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_CACHE_DIR = Path(os.environ.get("CACHE_DIR", Path.home() / ".cache/lean_dojo"))
print(DEFAULT_DATA_PATH, DEFAULT_CACHE_DIR, sep='\n')
LEANDOJO_BENCHMARK_4_URL = (
    "https://zenodo.org/records/10929138/files/leandojo_benchmark_4.tar.gz?download=1"
)
DOWNLOADS = {
    LEANDOJO_BENCHMARK_4_URL: "84a75ce552b31731165d55542b1aaca9",
}

def check_md5(filename: str, gt_hashcode: str) -> bool:
    """
    Check the MD5 of a file against the ground truth.
    """
    if not os.path.exists(filename):
        return False
    # The file could be large.
    # See https://stackoverflow.com/questions/48122798/oserror-errno-22-invalid-argument-when-reading-a-huge-file.
    inp = open(filename, "rb")
    hasher = md5()
    while True:
        block = inp.read(64 * (1 << 20))
        if not block:
            break
        hasher.update(block)
    return hasher.hexdigest() == gt_hashcode

def rename_target_as_backup(target_path, backup_name=None, override_backup=False):
    """
    Post-condition: target_path does not exist**
    **unless backup is provided, another backup already exists, and override_backup is True.

    """
    if not os.path.exists(target_path):
        logger.info("Target does not exist.")
        return

    # target exists...
    if backup_name is None:
        # no backup name provided, remove the target
        logger.info("No backup requested, deleting original directory.")
        shutil.rmtree(target_path)
        return
    
    # target exists and backup is requested...
    # - get the backup path
    if isinstance(backup_name, Path) or '/' in backup_name:
        # backup name is a path, use it as is
        backup_path = Path(backup_name)
    else:
        backup_path = Path(target_path).parent / backup_name

    # override / error out if backup exists
    if os.path.exists(backup_path):
        if override_backup:
            logger.info(f"Overriding existing backup at {backup_path}")
            shutil.rmtree(backup_path)
        else:
            raise RuntimeError(f"Backup folder {backup_name} already exists.")
    logger.info(f"Creating backup at {backup_path}")
    shutil.move(target_path, backup_path)
    
def download_benchmark(
    data_path, 
    data_backup=None, 
    cache_path=DEFAULT_CACHE_DIR, 
    cache_backup=None,
    override_backup=False
):
    # handle keeping the old data/renaming it
    if os.path.exists(data_path):
        rename_target_as_backup(data_path, data_backup, override_backup)
    os.mkdir(data_path)
    
    for url, hashcode in DOWNLOADS.items():
        logger.info(f"Downloading {url}")
        path = f"{data_path}/{os.path.basename(url)}"
        os.system(f"wget {url} -O {path}")
        if not check_md5(path, hashcode):
            raise RuntimeError(f"MD5 of {path} does not match the ground truth.")

        logger.info(f"Extracting {path}")
        os.system(f"tar -xf {path} -C {data_path}")

        logger.info(f"Removing {path}")
        os.remove(path)

    logger.info("Done downloading data!")

def download_cache(
    data_path=DEFAULT_DATA_PATH,
    cache_path=DEFAULT_CACHE_DIR, 
    cache_backup=None,
    override_backup=False
):
    logger.info("Downloading cache...")

    # move the existing cache to a backup
    if os.path.exists(cache_path):
        rename_target_as_backup(cache_path, cache_backup, override_backup)
    os.mkdir(cache_path)
    
    # populate cache by triggering a check
    data_file = data_path / "leandojo_benchmark_4" / "random" / "test.json"
    with open(data_file) as f:
        data = json.load(f)
    theorem = data[0] 
    repo = LeanGitRepo(url=theorem["url"], commit=theorem["commit"])
    assert is_available_in_cache(repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", '-d', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--cache_path", '-c', type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--data_backup", '-db')
    parser.add_argument("--cache_backup", '-cb')
    parser.add_argument("--force", '-f', action='store_true')
    args = parser.parse_args()
    logger.info(args)

    download_benchmark(
        Path(args.data_path),
        cache_path=Path(args.cache_path), 
        data_backup=args.data_backup, 
        cache_backup=args.cache_backup,
        override_backup=args.force
    )

if __name__ == "__main__":
    main()
