import re
import tomllib
from pathlib import Path

from packaging.version import Version


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MINIMUM_SAFE_SEMANTIC_ROUTER_VERSION = Version("0.1.15")


def _assert_safe_version(version: str, source: str) -> None:
    assert Version(version) >= MINIMUM_SAFE_SEMANTIC_ROUTER_VERSION, (
        f"{source} must not install a semantic-router version affected by "
        "CVE-2026-42208"
    )


def test_semantic_router_security_version_is_consistent() -> None:
    install_script = (REPOSITORY_ROOT / "docker/install_auto_router.sh").read_text()
    script_match = re.search(r"semantic_router==(\S+)\s+--no-deps", install_script)
    assert script_match is not None
    _assert_safe_version(script_match.group(1), "docker/install_auto_router.sh")

    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    pyproject_version = pyproject["tool"]["poetry"]["dependencies"][
        "semantic-router"
    ]["version"]
    _assert_safe_version(pyproject_version, "pyproject.toml")

    lock_content = (REPOSITORY_ROOT / "poetry.lock").read_text()
    lock_match = re.search(
        r'\[\[package\]\]\nname = "semantic-router"\nversion = "([^"]+)"',
        lock_content,
    )
    assert lock_match is not None
    _assert_safe_version(lock_match.group(1), "poetry.lock")
