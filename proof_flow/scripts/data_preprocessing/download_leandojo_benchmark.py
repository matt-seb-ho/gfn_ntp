"""Script to download LeanDojo Benchmark and LeanDojo Benchmark 4 into `./data`."""

import argparse
import os
from hashlib import md5

from loguru import logger

from proof_flow.src.utils import repo_root

# lean dojo benchmark 4 url: https://zenodo.org/records/12740403
LEANDOJO_BENCHMARK_4_URL = (
    "https://zenodo.org/records/12740403/files/leandojo_benchmark_4.tar.gz?download=1"
)
DOWNLOADS = {
    LEANDOJO_BENCHMARK_4_URL: "25e1ee60cd8925b9d2e8673ddcc34b4c",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    default_data_path = str(repo_root() / "data")
    parser.add_argument("--data-path", type=str, default=default_data_path)
    args = parser.parse_args()
    logger.info(args)

    confirm_msg = (
        f"data_path: {args.data_path} (will download into f'{{data_path}}/leandojo_benchmark'').\n"
        "Confirm [y/n]: "
    )
    print(confirm_msg, end="")
    res = input()
    if res != "y":
        return

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    for url, hashcode in DOWNLOADS.items():
        logger.info(f"Downloading {url}")
        path = f"{args.data_path}/{os.path.basename(url)}"
        os.system(f"wget {url} -O {path}")
        if not check_md5(path, hashcode):
            raise RuntimeError(f"MD5 of {path} does not match the ground truth.")

        logger.info(f"Extracting {path}")
        os.system(f"tar -xf {path} -C {args.data_path}")

        logger.info(f"Removing {path}")
        os.remove(path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
