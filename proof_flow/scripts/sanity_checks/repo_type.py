import os
import re
import tempfile
import urllib
from contextlib import contextmanager
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Generator, Optional, Union

from loguru import logger

from proof_flow.src.utils import prepare_environment_for_lean_dojo

prepare_environment_for_lean_dojo()

from lean_dojo import LeanGitRepo # isort: skip

_SSH_TO_HTTPS_REGEX = re.compile(r"^git@github\.com:(.+)/(.+)(?:\.git)?$")

REPO_CACHE_PREFIX = "repos"

TMP_DIR = Path(os.environ["TMP_DIR"]).absolute() if "TMP_DIR" in os.environ else None


class RepoType(Enum):
    GITHUB = 0
    REMOTE = 1  # Remote but not GitHub.
    LOCAL = 2

@cache
def url_exists(url: str) -> bool:
    """Return True if the URL ``url`` exists, using the GITHUB_ACCESS_TOKEN for authentication if provided."""
    try:
        request = urllib.request.Request(url)  # type: ignore
        gh_token = os.getenv("GITHUB_ACCESS_TOKEN")
        if gh_token is not None:
            request.add_header("Authorization", f"token {gh_token}")
        with urllib.request.urlopen(request) as _:  # type: ignore
            return True
    except urllib.error.HTTPError:  # type: ignore
        return False

@contextmanager
def working_directory(
    path: Optional[Union[str, Path]] = None
) -> Generator[Path, None, None]:
    """Context manager setting the current working directory (CWD) to ``path`` (or a temporary directory if ``path`` is None).

    The original CWD is restored after the context manager exits.

    Args:
        path (Optional[Union[str, Path]], optional): The desired CWD. Defaults to None.

    Yields:
        Generator[Path, None, None]: A ``Path`` object representing the CWD.
    """
    origin = Path.cwd()
    if path is None:
        tmp_dir = tempfile.TemporaryDirectory(dir=TMP_DIR)
        path = tmp_dir.__enter__()
        is_temporary = True
    else:
        is_temporary = False

    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    os.chdir(path)

    try:
        yield path
    finally:
        os.chdir(origin)
        if is_temporary:
            tmp_dir.__exit__(None, None, None)

@cache
def is_git_repo(path: Path) -> bool:
    """Check if ``path`` is a Git repo."""
    with working_directory(path):
        return (
            os.system("git rev-parse --is-inside-work-tree 1>/dev/null 2>/dev/null")
            == 0
        )

def normalize_url(url: str, repo_type: RepoType = RepoType.GITHUB) -> str:
    if repo_type == RepoType.LOCAL:  # Convert to absolute path if local.
        return os.path.abspath(url)
    # Remove trailing `/`.
    return _URL_REGEX.fullmatch(url)["url"]  # type: ignore


def get_repo_type(url: str) -> Optional[RepoType]:
    """Get the type of the repository.

    Args:
        url (str): The URL of the repository.
    Returns:
        Optional[str]: The type of the repository (None if the repo cannot be found).
    """
    m = _SSH_TO_HTTPS_REGEX.match(url)
    url = f"https://github.com/{m.group(1)}/{m.group(2)}" if m else url
    parsed_url = urllib.parse.urlparse(url)  # type: ignore
    if parsed_url.scheme in ["http", "https"]:
        # Case 1 - GitHub URL.
        if "github.com" in url:
            if not url.startswith("https://"):
                logger.warning(f"{url} should start with https://")
                return None
            else:
                return RepoType.GITHUB
        # Case 2 - remote URL.
        elif url_exists(url):  # Not check whether it is a git URL
            return RepoType.REMOTE
    # Case 3 - local path
    elif is_git_repo(Path(parsed_url.path)):
        return RepoType.LOCAL
    logger.warning(f"{url} is not a valid URL")
    return None

if __name__ == "__main__":
    """
    {'url': 'https://github.com/leanprover-community/mathlib4',
    'commit': 'fe4454af900584467d21f4fd4fe951d29d9332a7',
    """
    url = "https://github.com/leanprover-community/mathlib4"
    commit = "fe4454af900584467d21f4fd4fe951d29d9332a7"
    # repo = LeanGitRepo(url, commit)
    # repo.repo_t
    print(get_repo_type(url))
