from __future__ import annotations

import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path


def test_package_metadata_version_matches_pyproject() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert metadata.version("fitpredict") == pyproject["project"]["version"]


def test_version_check_accepts_matching_tag() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_version.py", "--tag", "v1.0.1", "--skip-artifacts"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "fitpredict version metadata OK: 1.0.1"


def test_version_check_rejects_mismatched_tag() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_version.py", "--tag", "v9.9.9"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Version mismatch" in result.stderr
