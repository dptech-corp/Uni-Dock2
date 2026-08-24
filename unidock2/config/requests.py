"""Runtime inputs after defaults, YAML and explicit CLI values are merged."""

from dataclasses import dataclass

from unidock2.config.models import UnidockConfig


@dataclass(frozen=True)
class ResolvedDockingRequest:
    receptor_file_name: str
    ligand_sdf_file_name_list: tuple[str, ...]
    target_center: tuple[float, float, float]
    root_temp_dir_name: str
    docking_pose_sdf_file_name: str
    remove_temp_dir: bool
    config: UnidockConfig


@dataclass(frozen=True)
class ResolvedProteinPrepRequest:
    receptor_file_name: str
    root_temp_dir_name: str
    receptor_dms_file_name: str
    remove_temp_dir: bool
    config: UnidockConfig
