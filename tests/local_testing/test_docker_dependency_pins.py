import re
import tomllib
from pathlib import Path

from packaging.version import Version


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MINIMUM_SAFE_SEMANTIC_ROUTER_VERSION = Version("0.1.15")
MINIMUM_SAFE_NODE_TAR_VERSION = Version("7.5.21")


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
    pyproject_version = pyproject["tool"]["poetry"]["dependencies"]["semantic-router"][
        "version"
    ]
    _assert_safe_version(pyproject_version, "pyproject.toml")

    lock_content = (REPOSITORY_ROOT / "poetry.lock").read_text()
    lock_match = re.search(
        r'\[\[package\]\]\nname = "semantic-router"\nversion = "([^"]+)"',
        lock_content,
    )
    assert lock_match is not None
    _assert_safe_version(lock_match.group(1), "poetry.lock")

    non_root_dockerfile = (REPOSITORY_ROOT / "docker/Dockerfile.non_root").read_text()
    non_root_versions = re.findall(r"semantic_router==([0-9.]+)", non_root_dockerfile)
    assert non_root_versions
    for version in non_root_versions:
        _assert_safe_version(version, "docker/Dockerfile.non_root")


def test_node_tar_security_version_is_consistent() -> None:
    dockerfiles = (
        "Dockerfile",
        "docker/Dockerfile.custom_ui",
        "docker/Dockerfile.database",
        "docker/Dockerfile.dev",
        "docker/Dockerfile.non_root",
    )

    for dockerfile in dockerfiles:
        content = (REPOSITORY_ROOT / dockerfile).read_text()
        installed_versions = re.findall(r"tar@([0-9.]+)", content)
        metadata_versions = re.findall(r'"tar": "\^([0-9.]+)"', content)

        assert installed_versions, f"{dockerfile} must explicitly pin node-tar"
        assert metadata_versions, f"{dockerfile} must patch npm's node-tar metadata"
        for version in (*installed_versions, *metadata_versions):
            assert Version(version) >= MINIMUM_SAFE_NODE_TAR_VERSION, (
                f"{dockerfile} must not install a node-tar version affected by "
                "GHSA-r292-9mhp-454m"
            )
