"""YAML adapter for the typed Uni-Dock2 configuration models."""

from pathlib import Path

from pydantic import ValidationError
import yaml

from unidock2.config import UnidockConfig as _UnidockConfig

DEFAULT_CONFIG_FILE_NAME = "unidock2_config.yaml"


class _IndentedSafeDumper(yaml.SafeDumper):
    """Indent sequence items below their mapping key for a readable template."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, indentless=False)


class _FlowStyleList(list):
    """Mark a simple list for compact YAML flow-style rendering."""


def _represent_flow_style_list(dumper, value):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


_IndentedSafeDumper.add_representer(_FlowStyleList, _represent_flow_style_list)


def _render_field(field_name, value, description=None):
    if isinstance(value, list) and all(not isinstance(item, (dict, list, tuple)) for item in value):
        value = _FlowStyleList(value)

    field_yaml = yaml.dump(
        {field_name: value},
        Dumper=_IndentedSafeDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).rstrip()
    lines = field_yaml.splitlines()
    if description:
        inline_comment = " ".join(description.split())
        lines[0] = f"{lines[0]}  # {inline_comment}"
    return "\n".join(lines)


def _serialized_field_name(field_name, field):
    return field.serialization_alias or field.alias or field_name


def render_default_config_yaml() -> str:
    """Render the Pydantic defaults as an annotated, round-trippable YAML template."""
    config = _UnidockConfig()
    serialized_config = config.model_dump(by_alias=True)
    lines = [
        "# Uni-Dock2 default docking configuration.",
        "# Replace null input paths before running docking.",
        "# Run with: unidock2 docking -cf <this-file>",
        "# Explicit command-line values override values in this file.",
    ]

    for section_name, section_field in type(config).model_fields.items():
        section = getattr(config, section_name)
        section_alias = _serialized_field_name(section_name, section_field)
        serialized_section = serialized_config[section_alias]
        section_description = (type(section).__doc__ or "").strip()

        lines.append("")
        if section_description:
            lines.append(f"# {section_description}")
        lines.append(f"{section_alias}:")

        for field_name, field in type(section).model_fields.items():
            field_alias = _serialized_field_name(field_name, field)
            field_yaml = _render_field(
                field_alias,
                serialized_section[field_alias],
                field.description,
            )
            lines.extend(f"  {line}" for line in field_yaml.splitlines())

    return "\n".join(lines) + "\n"


def dump_default_config_yaml(output_file: str | Path = DEFAULT_CONFIG_FILE_NAME) -> Path:
    """Write the annotated default configuration and return its absolute path."""
    output_path = Path(output_file).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path = output_path.resolve()
    try:
        with output_path.open("x", encoding="utf-8") as file:
            file.write(render_default_config_yaml())
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite existing configuration file: {output_path}") from error
    return output_path


def read_unidock_params_from_yaml(yaml_file: str) -> _UnidockConfig:
    """Read and validate Uni-Dock2 parameters from a YAML file."""
    with open(yaml_file, encoding="utf-8") as file:
        params = yaml.safe_load(file)

    if params is None:
        params = {}

    try:
        return _UnidockConfig.from_dict(params)
    except ValidationError as error:
        print(f"Configuration Error:\n{error.json(indent=2)}")
        raise
