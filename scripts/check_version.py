"""Check package version metadata against release inputs and build artifacts."""

from __future__ import annotations

import argparse
import re
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
TAG_RE = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)$")


def _project_version() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        data = tomllib.load(file)
    version = data["project"]["version"]
    if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
        raise SystemExit(f"Invalid project.version in pyproject.toml: {version!r}")
    return version


def _metadata_version(metadata_text: str, artifact: Path) -> str:
    metadata = Parser().parsestr(metadata_text)
    name = metadata.get("Name")
    version = metadata.get("Version")
    if name != "fitpredict":
        raise SystemExit(f"{artifact}: expected Name fitpredict, got {name!r}")
    if not version:
        raise SystemExit(f"{artifact}: missing Version metadata")
    return version


def _wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as wheel:
        metadata_names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise SystemExit(f"{path}: expected exactly one METADATA file")
        return _metadata_version(wheel.read(metadata_names[0]).decode(), path)


def _sdist_version(path: Path) -> str:
    with tarfile.open(path) as sdist:
        metadata_names = [
            name for name in sdist.getnames() if name.count("/") == 1 and name.endswith("/PKG-INFO")
        ]
        if len(metadata_names) != 1:
            raise SystemExit(f"{path}: expected exactly one top-level PKG-INFO file")
        member = sdist.extractfile(metadata_names[0])
        if member is None:
            raise SystemExit(f"{path}: cannot read PKG-INFO")
        return _metadata_version(member.read().decode(), path)


def _artifact_versions(dist_dir: Path) -> dict[Path, str]:
    if not dist_dir.exists():
        raise SystemExit(f"Distribution directory does not exist: {dist_dir}")

    artifacts = sorted([*dist_dir.glob("*.whl"), *dist_dir.glob("*.tar.gz")])
    if not artifacts:
        raise SystemExit(f"No wheel or sdist artifacts found in {dist_dir}")

    versions: dict[Path, str] = {}
    for artifact in artifacts:
        if artifact.suffix == ".whl":
            versions[artifact] = _wheel_version(artifact)
        elif artifact.name.endswith(".tar.gz"):
            versions[artifact] = _sdist_version(artifact)
    return versions


def _version_from_tag(tag: str) -> str:
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise SystemExit(f"Release tag must look like v1.0.1, got {tag!r}")
    return match.group("version")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Release tag such as v1.0.1")
    parser.add_argument("--expected-version", help="Expected package version")
    parser.add_argument("--dist-dir", default="dist", help="Directory with built artifacts")
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Only check pyproject/tag/input version agreement",
    )
    args = parser.parse_args()

    expected_versions = [_project_version()]
    if args.tag:
        expected_versions.append(_version_from_tag(args.tag))
    if args.expected_version:
        expected_versions.append(args.expected_version)

    expected = expected_versions[0]
    mismatches = [version for version in expected_versions[1:] if version != expected]
    if mismatches:
        raise SystemExit(f"Version mismatch: expected {expected}, got {expected_versions[1:]}")

    if not args.skip_artifacts:
        for artifact, version in _artifact_versions(PROJECT_ROOT / args.dist_dir).items():
            if version != expected:
                raise SystemExit(f"{artifact}: expected version {expected}, got {version}")

    print(f"fitpredict version metadata OK: {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
