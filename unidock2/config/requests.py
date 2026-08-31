"""Runtime inputs after defaults, YAML and explicit CLI values are merged."""

from dataclasses import dataclass

from unidock2.config.models import UnidockConfig


LIGAND_SOURCE_SDF_FILES = "sdf_files"
LIGAND_SOURCE_UD2LIG = "ud2lig"


@dataclass(frozen=True)
class ResolvedDockingRequest:
    receptor_file_name: str
    ligand_source: str
    ligand_sdf_file_name_list: tuple[str, ...]
    ud2lig_dir: str | None
    target_center: tuple[float, float, float]
    root_temp_dir_name: str
    docking_pose_sdf_file_name: str
    remove_temp_dir: bool
    config: UnidockConfig


@dataclass(frozen=True)
class ResolvedPrepareProteinRequest:
    receptor_file_name: str
    root_temp_dir_name: str
    receptor_dms_file_name: str
    remove_temp_dir: bool
    config: UnidockConfig


@dataclass(frozen=True)
class ResolvedPrepareLigandsRequest:
    ligand_sdf_file_name_list: tuple[str, ...]
    output_ud2lig_dir: str
    root_temp_dir_name: str
    remove_temp_dir: bool
    config: UnidockConfig
