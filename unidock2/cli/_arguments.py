"""Build argparse options from Pydantic configuration metadata."""

import argparse
from types import UnionType
from typing import Union, get_args, get_origin

from unidock2.config import UnidockConfig


def _iter_cli_fields(command):
    defaults = UnidockConfig()
    for _, field_name, field, business_default in defaults.iter_flat_fields():
        extra = field.json_schema_extra or {}
        cli_info = extra.get("cli") if isinstance(extra, dict) else None
        if cli_info and command in cli_info["commands"]:
            yield field_name, field, business_default, cli_info


def iter_cli_config_field_names(command):
    """Yield schema fields exposed by a CLI command."""
    for field_name, _, _, _ in _iter_cli_fields(command):
        yield field_name


def _argument_type(annotation):
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        annotation = next(item for item in get_args(annotation) if item is not type(None))
        origin = get_origin(annotation)
    if origin is list:
        return get_args(annotation)[0]
    return annotation


def add_config_arguments(parser, command, *, with_config_file=True):
    """Add command-specific arguments whose metadata lives in the schema."""
    for field_name, field, business_default, cli_info in _iter_cli_fields(command):
        argument_type = _argument_type(field.annotation)
        options = {
            "default": None,
            "dest": field_name,
            "help": f"{field.description} (default: {business_default!r})",
        }
        if argument_type is bool:
            options["action"] = argparse.BooleanOptionalAction
        else:
            options["type"] = argument_type
        if "nargs" in cli_info:
            options["nargs"] = cli_info["nargs"]
        if "metavar" in cli_info:
            options["metavar"] = cli_info["metavar"]
        parser.add_argument(*cli_info["flags"], **options)

    if with_config_file:
        parser.add_argument(
            "-cf",
            "--config",
            dest="configurations",
            metavar="FILE",
            default=None,
            help="Uni-Dock2 configuration YAML file",
        )
