"""Configuration errors raised before experiment execution."""


class ConfigError(ValueError):
    """Raised when a user config cannot be loaded or represented by the schema."""
