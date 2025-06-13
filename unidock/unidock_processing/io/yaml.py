from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import yaml

@dataclass
class RequiredConfig:
    receptor: str = None
    ligand: str = None
    ligand_batch: str = None
    center: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )

@dataclass
class AdvancedConfig:
    exhaustiveness: int = 512
    randomize: bool = True
    mc_steps: int = 40
    opt_steps: int = -1
    refine_steps: int = 5
    num_pose: int = 10
    rmsd_limit: float = 1.0
    energy_range: float = 5.0
    seed: int = 1234567
    use_tor_lib: bool = False

@dataclass
class HardwareConfig:
    gpu_device_id: int = 0

@dataclass
class SettingsConfig:
    size_x: float = 30.0
    size_y: float = 30.0
    size_z: float = 30.0
    task: str = "screen"
    search_mode: str = "balance"

@dataclass
class PreprocessingConfig:
    template_docking: bool = False
    reference_sdf_file_name: Optional[str] = None
    core_atom_mapping_dict_list: Optional[List[Dict[str, Any]]] = None
    covalent_ligand: bool = False
    covalent_residue_atom_info_list: Optional[List[Any]] = None
    preserve_receptor_hydrogen: bool = False
    remove_temp_files: bool = True
    working_dir_name: str = "./"

@dataclass
class UnidockConfig:
    required: RequiredConfig = field(default_factory=RequiredConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    settings: SettingsConfig = field(default_factory=SettingsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnidockConfig":
        """Create UnidockConfig from a dictionary."""
        config = cls()

        if "Required" in data:
            config.required = RequiredConfig(**data["Required"])

        if "Advanced" in data:
            config.advanced = AdvancedConfig(**data["Advanced"])

        if "Hardware" in data:
            config.hardware = HardwareConfig(**data["Hardware"])

        if "Settings" in data:
            config.settings = SettingsConfig(**data["Settings"])

        if "Preprocessing" in data:
            config.preprocessing = PreprocessingConfig(**data["Preprocessing"])

        return config

    def to_protocol_kwargs(self) -> Dict[str, Any]:
        kwargs_dict = {
            "receptor": self.required.receptor,
            "ligand": self.required.ligand,
            "ligand_batch": self.required.ligand_batch,
            "center": self.required.center,
            "template_docking": self.preprocessing.template_docking,
            "reference_sdf_file_name": self.preprocessing.reference_sdf_file_name,
            "core_atom_mapping_dict_list": \
                self.preprocessing.core_atom_mapping_dict_list,
            "covalent_ligand": self.preprocessing.covalent_ligand,
            "covalent_residue_atom_info_list": \
                self.preprocessing.covalent_residue_atom_info_list,
            "preserve_receptor_hydrogen": self.preprocessing.preserve_receptor_hydrogen,
            "remove_temp_files": self.preprocessing.remove_temp_files,
            "working_dir_name": self.preprocessing.working_dir_name,
            "gpu_device_id": self.hardware.gpu_device_id,
            "box_size": (self.settings.size_x, self.settings.size_y,
                         self.settings.size_z),
            "task": self.settings.task,
            "search_mode": self.settings.search_mode,
            "exhaustiveness": self.advanced.exhaustiveness,
            "randomize": self.advanced.randomize,
            "mc_steps": self.advanced.mc_steps,
            "opt_steps": self.advanced.opt_steps,
            "refine_steps": self.advanced.refine_steps,
            "num_pose": self.advanced.num_pose,
            "rmsd_limit": self.advanced.rmsd_limit,
            "energy_range": self.advanced.energy_range,
            "seed": self.advanced.seed,
            "use_tor_lib": self.advanced.use_tor_lib,
        }
        return kwargs_dict

def read_unidock_params_from_yaml(yaml_file: str) -> UnidockConfig:
    """
    Read Unidock parameters from a yaml file and convert to UnidockConfig dataclass.

    Args:
        yaml_file: Path to the yaml file

    Returns:
        UnidockConfig object
    """
    with open(yaml_file, "r") as f:
        params = yaml.safe_load(f)

    return UnidockConfig.from_dict(params)
