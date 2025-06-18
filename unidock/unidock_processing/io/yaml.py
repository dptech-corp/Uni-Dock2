from typing import Dict, Any, Optional, ClassVar, List, Tuple, Type
from dataclasses import dataclass, field
import yaml

@dataclass
class RequiredConfig:
    receptor: str = None
    ligand: str = None
    ligand_batch: str = None
    center: Tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0)
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
    box_size: Tuple[float, float, float] = field(
        default_factory=lambda: (30.0, 30.0, 30.0)
    )
    task: str = 'screen'
    search_mode: str = 'balance'

@dataclass
class PreprocessingConfig:
    template_docking: bool = False
    reference_sdf_file_name: Optional[str] = None
    core_atom_mapping_dict_list: Optional[List[Dict[str, Any]]] = None
    covalent_ligand: bool = False
    covalent_residue_atom_info_list: Optional[List[Any]] = None
    preserve_receptor_hydrogen: bool = False
    temp_dir_name: str = '/tmp'
    output_receptor_dms_file_name: str = 'receptor_parameterized.dms'
    output_docking_pose_sdf_file_name: str = 'unidock2_pose.sdf'

CONFIG_MAPPING: ClassVar[List[Tuple[str, str, Type, List[str]]]] = [
    ('Required', 'required', RequiredConfig,
    ['receptor', 'ligand', 'ligand_batch', 'center']),
    ('Advanced', 'advanced', AdvancedConfig, [
    'exhaustiveness', 'randomize', 'mc_steps', 'opt_steps',
    'refine_steps', 'num_pose', 'rmsd_limit', 'energy_range',
    'seed', 'use_tor_lib'
    ]),
    ('Hardware', 'hardware', HardwareConfig, ['gpu_device_id']),
    ('Settings', 'settings', SettingsConfig, ['box_size', 'task', 'search_mode']),
    ('Preprocessing', 'preprocessing', PreprocessingConfig, [
    'template_docking', 'reference_sdf_file_name',
    'core_atom_mapping_dict_list', 'covalent_ligand',
    'covalent_residue_atom_info_list', 'preserve_receptor_hydrogen',
    'temp_dir_name', 'output_receptor_dms_file_name',
    'output_docking_pose_sdf_file_name'
    ])
]

def config_sections(cls):
    """automatically generate config decorators"""
    for _, attr_name, config_cls, _ in CONFIG_MAPPING:
        setattr(cls, attr_name, field(default_factory=config_cls))

        if not hasattr(cls, '__annotations__'):
            cls.__annotations__ = {}
        cls.__annotations__[attr_name] = config_cls

    return dataclass(cls)

@config_sections
class UnidockConfig:

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnidockConfig':
        """Create UnidockConfig from a dictionary."""
        config = cls()
        for dict_key, attr_name, config_cls, _ in CONFIG_MAPPING:
            if dict_key in data:
                setattr(config, attr_name, config_cls(**data[dict_key]))

        return config

    def to_protocol_kwargs(self) -> Dict[str, Any]:
        kwargs_dict = {}
        for _, attr_name, _, field_names in CONFIG_MAPPING:
            config = getattr(self, attr_name)
            for field_name in field_names:
                kwargs_dict[field_name] = getattr(config, field_name)

        return kwargs_dict

def read_unidock_params_from_yaml(yaml_file: str) -> UnidockConfig:
    """
    Read Unidock parameters from a yaml file and convert to UnidockConfig dataclass.

    Args:
        yaml_file: Path to the yaml file

    Returns:
        UnidockConfig object
    """
    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)

    return UnidockConfig.from_dict(params)
