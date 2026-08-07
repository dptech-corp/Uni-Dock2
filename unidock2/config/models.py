"""Single source of truth for Python-facing Uni-Dock2 parameters."""

from collections.abc import Iterator, Mapping
from typing import Any, ClassVar
import warnings

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.fields import FieldInfo


class UnknownConfigurationWarning(UserWarning):
    """Warning emitted when an unsupported configuration key is ignored."""


def cli(*flags, commands, nargs=None, metavar=None):
    """Attach argparse metadata to a Pydantic field."""
    metadata = {"flags": flags, "commands": commands}
    if nargs is not None:
        metadata["nargs"] = nargs
    if metavar is not None:
        metadata["metavar"] = metavar
    return {"cli": metadata}


class _ConfigurationSection(BaseModel):
    """Preserve ignore-unknown behavior while making ignored keys visible."""

    model_config = ConfigDict(extra="ignore")
    yaml_section_name: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def warn_about_unknown_fields(cls, data):
        if not isinstance(data, Mapping):
            return data

        known_fields = set(cls.model_fields)
        for field_name in data:
            if field_name not in known_fields:
                warnings.warn(
                    f"Unknown configuration field '{cls.yaml_section_name}.{field_name}' will be ignored.",
                    UnknownConfigurationWarning,
                    stacklevel=4,
                )
        return data


class RequiredConfig(_ConfigurationSection):
    yaml_section_name = "Required"

    receptor: str | None = Field(
        default=None,
        description="Receptor structure file in PDB or DMS format",
        json_schema_extra=cli(
            "-r",
            "--receptor",
            commands=("docking", "protein_prep"),
        ),
    )
    ligand: str | None = Field(
        default=None,
        description="Single ligand structure file in SDF format",
        json_schema_extra=cli("-l", "--ligand", commands=("docking",)),
    )
    ligand_batch: str | None = Field(
        default=None,
        description="Text file containing ligand SDF file paths",
        json_schema_extra=cli("-lb", "--ligand_batch", commands=("docking",)),
    )
    center: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Docking box center coordinates",
        json_schema_extra=cli(
            "-c",
            "--center",
            commands=("docking",),
            nargs=3,
            metavar=("center_x", "center_y", "center_z"),
        ),
    )

    @field_validator("center")
    @classmethod
    def validate_center(cls, value):
        if len(value) != 3:
            raise ValueError("Center requires 3 elements")
        return value


class AdvancedConfig(_ConfigurationSection):
    yaml_section_name = "Advanced"

    exhaustiveness: int = Field(
        default=512,
        description="Number of independent search tasks",
        json_schema_extra=cli("-e", "--exhaustiveness", commands=("docking",)),
    )
    randomize: bool = True
    mc_steps: int = 40
    opt_steps: int = -1
    refine_steps: int = 5
    num_pose: int = 10
    rmsd_limit: float = 1.0
    energy_range: float = 5.0
    seed: int = Field(
        default=1234567,
        description="Random seed",
        json_schema_extra=cli("--seed", commands=("docking",)),
    )
    use_tor_lib: bool = False
    energy_decomp: bool = False


class HardwareConfig(_ConfigurationSection):
    yaml_section_name = "Hardware"

    n_cpu: int | None = None
    gpu_device_id: int = Field(
        default=0,
        description="GPU device index",
        json_schema_extra=cli("--gpu_device_id", commands=("docking",)),
    )


class SettingsConfig(_ConfigurationSection):
    yaml_section_name = "Settings"

    box_size: list[float] = Field(
        default_factory=lambda: [30.0, 30.0, 30.0],
        description="Docking box dimensions",
        json_schema_extra=cli(
            "--box_size",
            commands=("docking",),
            nargs=3,
            metavar=("size_x", "size_y", "size_z"),
        ),
    )
    task: str = "screen"
    search_mode: str = Field(
        default="balance",
        description="Native engine search mode",
        json_schema_extra=cli("--search_mode", commands=("docking",)),
    )

    @field_validator("box_size")
    @classmethod
    def validate_box_size(cls, value):
        if len(value) != 3:
            raise ValueError("Box Size requires 3 elements")
        return value


class PreprocessingConfig(_ConfigurationSection):
    yaml_section_name = "Preprocessing"

    construct_ff: bool = False
    template_docking: bool = False
    reference_sdf_file_name: str | None = None
    compute_center: bool = True
    core_atom_mapping_dict_list: list[dict[Any, Any] | None] | None = None
    covalent_ligand: bool = False
    covalent_residue_atom_info_list: list[Any] | None = None
    preserve_receptor_hydrogen: bool = False
    temp_dir_name: str = "/tmp"
    engine_checkpoint: bool = False
    output_receptor_dms_file_name: str = Field(
        default="receptor_parameterized.dms",
        description="Output receptor DMS file name",
        json_schema_extra=cli(
            "-o",
            "--output_receptor_dms_file_name",
            commands=("protein_prep",),
        ),
    )
    output_docking_pose_sdf_file_name: str = Field(
        default="unidock2_pose.sdf",
        description="Output docking pose SDF file name",
        json_schema_extra=cli(
            "-o",
            "--output_docking_pose_sdf_file_name",
            commands=("docking",),
        ),
    )


class UnidockConfig(BaseModel):
    """Complete Uni-Dock2 configuration grouped by YAML section."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    required: RequiredConfig = Field(default_factory=RequiredConfig, alias="Required")
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig, alias="Advanced")
    hardware: HardwareConfig = Field(default_factory=HardwareConfig, alias="Hardware")
    settings: SettingsConfig = Field(default_factory=SettingsConfig, alias="Settings")
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        alias="Preprocessing",
    )

    @model_validator(mode="before")
    @classmethod
    def warn_about_unknown_sections(cls, data):
        if not isinstance(data, Mapping):
            return data

        known_names = set(cls.model_fields)
        known_aliases = {field.alias for field in cls.model_fields.values() if field.alias is not None}
        for section_name in data:
            if section_name not in known_names and section_name not in known_aliases:
                warnings.warn(
                    f"Unknown configuration section '{section_name}' will be ignored.",
                    UnknownConfigurationWarning,
                    stacklevel=4,
                )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnidockConfig":
        """Create a validated configuration from YAML-shaped data."""
        return cls.model_validate(data)

    def iter_flat_fields(self) -> Iterator[tuple[str, str, FieldInfo, Any]]:
        """Yield section, field name, metadata and value in schema order."""
        for section_name in type(self).model_fields:
            section = getattr(self, section_name)
            for field_name, field in type(section).model_fields.items():
                yield section_name, field_name, field, getattr(section, field_name)

    def _flat_field_locations(self) -> dict[str, str]:
        locations = {}
        for section_name, field_name, _, _ in self.iter_flat_fields():
            if field_name in locations:
                raise RuntimeError(f"Duplicate configuration field: {field_name}")
            locations[field_name] = section_name
        return locations

    def protocol_field_names(self) -> tuple[str, ...]:
        """Return flattened protocol field names in schema order."""
        return tuple(self._flat_field_locations())

    def with_overrides(self, **overrides: Any) -> "UnidockConfig":
        """Return a validated copy with flattened field overrides applied."""
        locations = self._flat_field_locations()
        unknown_fields = sorted(set(overrides) - set(locations))
        if unknown_fields:
            names = ", ".join(unknown_fields)
            raise TypeError(f"Unknown Uni-Dock2 configuration field(s): {names}")

        config_data = self.model_dump()
        for field_name, value in overrides.items():
            config_data[locations[field_name]][field_name] = value
        return type(self).model_validate(config_data)

    def to_protocol_kwargs(self) -> dict[str, Any]:
        """Flatten all sections for compatibility with existing callers."""
        return {field_name: value for _, field_name, _, value in self.iter_flat_fields()}
