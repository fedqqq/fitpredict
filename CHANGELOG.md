# Changelog

## 1.0.1 - 2026-09-18

### Added

- Stable root public API for `fit()`, `predict()`, config loading, config
  resolution, `ConfigError`, and `ExperimentConfig`.
- Runnable training and prediction examples with JSON/YAML configs.
- User docs for quickstart, config reference, binding sources, and release
  workflow.
- CI matrix for Python 3.11 and 3.12.
- GitHub Actions workflows for TestPyPI and PyPI publication through Trusted
  Publishing.

### Changed

- Cleaned package metadata and build configuration for wheel and sdist
  releases.
- Improved README so a new developer can install the package, write a model,
  run `fit()`, and run `predict()`.
- Improved user-facing configuration errors around loading, resolving,
  bindings, training, prediction, and optional logging integrations.

### Verified

- Lint, format check, type check, test suite, and syntax checks are covered in
  CI.
- TensorBoard and MLflow integrations are covered with test doubles, without a
  real external server.
- Clean wheel installation is supported outside the repository working tree.
