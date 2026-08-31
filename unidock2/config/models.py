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
    """Input structures and docking center."""

    yaml_section_name = "Required"

    receptor: str | None = Field(
        default=None,
        description=(
            "Receptor structure file in PDB or DMS format. "
            "A DMS file is treated as an already prepared receptor and skips protein preparation"
        ),
        json_schema_extra=cli(
            "-r",
            "--receptor",
            commands=("docking", "prepare_protein"),
        ),
    )
    ligand: str | None = Field(
        default=None,
        description=(
            "Ligand input: a single SDF file, a directory of SDF files, "
            "or a UD2LIG directory that contains manifest.json"
        ),
        json_schema_extra=cli("-l", "--ligand", commands=("docking", "prepare_ligands")),
    )
    ligand_batch: str | None = Field(
        default=None,
        description="Text file containing one ligand SDF file path per line",
        json_schema_extra=cli("-lb", "--ligand_batch", commands=("docking", "prepare_ligands")),
    )
    center: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Docking box center coordinates [x, y, z] in angstroms",
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
    """Docking search and pose-output controls."""

    yaml_section_name = "Advanced"

    exhaustiveness: int = Field(
        default=512,
        description="Number of independent Monte Carlo runs (roughly proportional to runtime)",
        json_schema_extra=cli("-e", "--exhaustiveness", commands=("docking",)),
    )
    randomize: bool = Field(
        default=True,
        description="Randomize the input pose before global search",
    )
    mc_steps: int = Field(
        default=40,
        description="Monte Carlo random-walk steps per run",
    )
    opt_steps: int = Field(
        default=-1,
        description="Optimization steps after each Monte Carlo step (-1 selects automatic)",
    )
    refine_steps: int = Field(
        default=5,
        description="Local refinement steps after pose clustering",
    )
    num_pose: int = Field(
        default=10,
        description="Maximum number of output poses per ligand",
    )
    rmsd_limit: float = Field(
        default=1.0,
        description="RMSD threshold in angstroms for pose clustering",
    )
    energy_range: float = Field(
        default=5.0,
        description="Energy window in kcal/mol for output poses",
    )
    seed: int = Field(
        default=1234567,
        description="Random seed for reproducibility",
        json_schema_extra=cli("--seed", commands=("docking",)),
    )
    bias: str = Field(
        default="no",
        description="Native bias mode: no, pos (position), or align",
    )
    bias_k: float = Field(
        default=0.1,
        description="Scaling coefficient applied to native bias potentials",
    )
    use_tor_lib: bool = Field(
        default=False,
        description="Use the torsion-angle library",
    )
    energy_decomp: bool = Field(
        default=False,
        description="Output per-atom intermolecular energy decomposition",
    )


class HardwareConfig(_ConfigurationSection):
    """CPU and GPU resource selection."""

    yaml_section_name = "Hardware"

    n_cpu: int | None = Field(
        default=None,
        description="Maximum CPU workers for ligand preprocessing (null uses available CPUs)",
    )
    gpu_device_id: int = Field(
        default=0,
        description="GPU device index to use",
        json_schema_extra=cli("--gpu_device_id", commands=("docking",)),
    )
    max_gpu_memory: int = Field(
        default=0,
        description="Maximum GPU memory in MB (0 uses the available-memory limit)",
    )


class SettingsConfig(_ConfigurationSection):
    """Docking box and native engine mode settings."""

    yaml_section_name = "Settings"

    box_size: list[float] = Field(
        default_factory=lambda: [30.0, 30.0, 30.0],
        description="Docking box dimensions [x, y, z] in angstroms",
        json_schema_extra=cli(
            "--box_size",
            commands=("docking",),
            nargs=3,
            metavar=("size_x", "size_y", "size_z"),
        ),
    )
    task: str = Field(
        default="screen",
        description="Native engine docking task type",
    )
    search_mode: str = Field(
        default="balance",
        description=("Search mode: fast, balance, detail, or free; non-free modes select preset search parameters"),
        json_schema_extra=cli("--search_mode", commands=("docking",)),
    )

    @field_validator("box_size")
    @classmethod
    def validate_box_size(cls, value):
        if len(value) != 3:
            raise ValueError("Box Size requires 3 elements")
        return value


class PreprocessingConfig(_ConfigurationSection):
    """Topology preparation, constrained docking, temporary files, and outputs."""

    yaml_section_name = "Preprocessing"

    construct_ff: bool = Field(
        default=False,
        description="Construct force-field atom types and bonded parameters for ligands",
        json_schema_extra=cli("--construct_ff", commands=("prepare_ligands",)),
    )
    template_docking: bool = Field(
        default=False,
        description="Enable template-constrained docking",
    )
    reference_sdf_file_name: str | None = Field(
        default=None,
        description="Reference ligand SDF file used by template-constrained docking",
    )
    compute_center: bool = Field(
        default=True,
        description="Recompute the docking center from the reference or first ligand in constrained modes",
    )
    core_atom_mapping_dict_list: list[dict[Any, Any] | None] | None = Field(
        default=None,
        description="Optional per-ligand core atom mappings for template-constrained docking",
    )
    covalent_ligand: bool = Field(
        default=False,
        description="Enable covalent docking",
    )
    covalent_residue_atom_info_list: list[Any] | None = Field(
        default=None,
        description="Three receptor atom descriptors defining the covalent anchor and bond",
    )
    preserve_receptor_hydrogen: bool = Field(
        default=False,
        description="Preserve receptor hydrogens during topology preparation",
    )
    temp_dir_name: str = Field(
        default="/tmp",
        description="Parent directory for temporary working directories",
    )
    engine_checkpoint: bool = Field(
        default=True,
        description=(
            "Write a UD2LIG library next to the pose SDF after ligand preparation. "
            "Skipped when docking from an existing UD2LIG, or for template/covalent jobs"
        ),
    )
    output_receptor_dms_file_name: str = Field(
        default="receptor_parameterized.dms",
        description="Output receptor DMS file name",
        json_schema_extra=cli(
            "-o",
            "--output_receptor_dms_file_name",
            commands=("prepare_protein",),
        ),
    )
    output_docking_pose_sdf_file_name: str = Field(
        default="unidock2_pose.sdf",
        description="Output docking pose SDF file name",
        json_schema_extra=cli(
            "-o",
            "--output",
            commands=("docking",),
            metavar="SDF",
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
