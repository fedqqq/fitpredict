# Release

This project publishes packages through GitHub Actions and PyPI Trusted
Publishing. No PyPI API token is required.

## TestPyPI

1. In TestPyPI, add a Trusted Publisher for this repository.
2. Use workflow `.github/workflows/publish-testpypi.yml`.
3. Run the workflow manually with the expected version, for example `1.0.1`.
4. The workflow builds `sdist` and `wheel`, checks artifact metadata, publishes
   to TestPyPI, and verifies installation from TestPyPI.

## PyPI

1. In PyPI, add a Trusted Publisher for this repository.
2. Configure the publisher for workflow `.github/workflows/publish-pypi.yml`
   and environment `pypi`.
3. Push a release tag matching the package version:

   ```bash
   git tag v1.0.1
   git push origin v1.0.1
   ```

4. The workflow runs the same quality gate as CI, builds the package, checks
   that `v1.0.1` matches package version `1.0.1`, creates a GitHub Release from
   `CHANGELOG.md`, and publishes to PyPI.

The package version has one source of truth: `[project].version` in
`pyproject.toml`. The release workflows fail if tag, package metadata, wheel,
and sdist versions do not match.
