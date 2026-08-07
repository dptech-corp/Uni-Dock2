"""Resolve defaults, YAML and explicit CLI values into runtime requests."""

import os

from unidock2.cli._arguments import iter_cli_config_field_names
from unidock2.config import (
    ResolvedDockingRequest,
    ResolvedProteinPrepRequest,
    UnidockConfig,
)
from unidock2.io.yaml import read_unidock_params_from_yaml


def load_config_from_args(args):
    """Load YAML when selected, otherwise return the schema defaults."""
    configurations = getattr(args, "configurations", None)
    if configurations:
        return read_unidock_params_from_yaml(configurations)
    return UnidockConfig()


def merge_cli_overrides(args, config, command):
    """Apply only explicitly supplied CLI values over a validated config."""
    overrides = {
        field_name: value
        for field_name in iter_cli_config_field_names(command)
        if (value := getattr(args, field_name, None)) is not None
    }
    return config.with_overrides(**overrides)


def resolve_docking_request(args, config=None):
    """Resolve docking inputs without invoking topology or GPU code."""
    if config is None:
        config = load_config_from_args(args)
    config = merge_cli_overrides(args, config, "docking")

    receptor = config.required.receptor
    if receptor is None:
        raise ValueError("Receptor file name not specified!")

    ligand_file_names = []
    if config.required.ligand is not None:
        ligand_file_names.append(os.path.abspath(config.required.ligand))

    if config.required.ligand_batch is not None:
        ligand_batch_file_name = os.path.abspath(config.required.ligand_batch)
        with open(ligand_batch_file_name, encoding="utf-8") as ligand_batch_file:
            for line in ligand_batch_file:
                ligand_file_name = line.strip()
                if ligand_file_name:
                    ligand_file_names.append(os.path.abspath(ligand_file_name))

    if not ligand_file_names:
        raise ValueError("Ligand SDF file input not found!")

    root_temp_dir_name = os.path.abspath(config.preprocessing.temp_dir_name)
    return ResolvedDockingRequest(
        receptor_file_name=os.path.abspath(receptor),
        ligand_sdf_file_name_list=tuple(ligand_file_names),
        target_center=tuple(config.required.center),
        root_temp_dir_name=root_temp_dir_name,
        docking_pose_sdf_file_name=os.path.abspath(config.preprocessing.output_docking_pose_sdf_file_name),
        remove_temp_dir=root_temp_dir_name == "/tmp",
        config=config,
    )


def resolve_protein_prep_request(args, config=None):
    """Resolve protein-preparation inputs without doing preparation work."""
    if config is None:
        config = load_config_from_args(args)
    config = merge_cli_overrides(args, config, "protein_prep")

    receptor = config.required.receptor
    if receptor is None:
        raise ValueError("Receptor file name not specified!")

    root_temp_dir_name = os.path.abspath(config.preprocessing.temp_dir_name)
    return ResolvedProteinPrepRequest(
        receptor_file_name=os.path.abspath(receptor),
        root_temp_dir_name=root_temp_dir_name,
        receptor_dms_file_name=os.path.abspath(config.preprocessing.output_receptor_dms_file_name),
        remove_temp_dir=root_temp_dir_name == "/tmp",
        config=config,
    )
