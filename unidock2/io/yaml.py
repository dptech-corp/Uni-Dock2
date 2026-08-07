"""YAML adapter for the typed Uni-Dock2 configuration models."""

from pydantic import ValidationError
import yaml

from unidock2.config.models import (
    AdvancedConfig,
    HardwareConfig,
    PreprocessingConfig,
    RequiredConfig,
    SettingsConfig,
    UnidockConfig,
    UnknownConfigurationWarning,
)

__all__ = [
    "AdvancedConfig",
    "HardwareConfig",
    "PreprocessingConfig",
    "RequiredConfig",
    "SettingsConfig",
    "UnidockConfig",
    "UnknownConfigurationWarning",
    "read_unidock_params_from_yaml",
]


def read_unidock_params_from_yaml(yaml_file: str) -> UnidockConfig:
    """Read and validate Uni-Dock2 parameters from a YAML file."""
    with open(yaml_file, encoding="utf-8") as file:
        params = yaml.safe_load(file)

    if params is None:
        params = {}

    try:
        return UnidockConfig.from_dict(params)
    except ValidationError as error:
        print(f"Configuration Error:\n{error.json(indent=2)}")
        raise
