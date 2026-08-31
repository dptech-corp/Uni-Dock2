"""Resolve defaults, YAML and explicit CLI values into runtime requests."""

import os

from unidock2.cli._arguments import iter_cli_config_field_names
from unidock2.config import (
    LIGAND_SOURCE_SDF_FILES,
    LIGAND_SOURCE_UD2LIG,
    ResolvedDockingRequest,
    ResolvedPrepareLigandsRequest,
    ResolvedPrepareProteinRequest,
    UnidockConfig,
)
from unidock2.io.ud2lig import (
    LIGAND_KIND_SDF_DIR,
    LIGAND_KIND_SDF_FILE,
    LIGAND_KIND_UD2LIG,
    classify_ligand_path,
    list_sdf_files,
    load_ud2lig_manifest,
    validate_ud2lig_against_config,
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


def _read_ligand_batch_file(ligand_batch_file_name):
    ligand_file_names = []
    with open(ligand_batch_file_name, encoding="utf-8") as ligand_batch_file:
        for line in ligand_batch_file:
            ligand_file_name = line.strip()
            if ligand_file_name:
                ligand_file_names.append(os.path.abspath(ligand_file_name))
    return ligand_file_names


def resolve_ligand_inputs(ligand, ligand_batch, *, allow_ud2lig):
    """Resolve -l / -lb into SDF paths or a UD2LIG directory."""
    sdf_file_names = []
    ud2lig_dir = None

    if ligand is not None:
        ligand_path = os.path.abspath(ligand)
        kind = classify_ligand_path(ligand_path)
        if kind == LIGAND_KIND_UD2LIG:
            if not allow_ud2lig:
                raise ValueError(
                    f"Ligand input {ligand_path!r} is already a UD2LIG directory "
                    "and cannot be prepared again."
                )
            ud2lig_dir = ligand_path
        elif kind == LIGAND_KIND_SDF_FILE:
            sdf_file_names.append(ligand_path)
        elif kind == LIGAND_KIND_SDF_DIR:
            sdf_file_names.extend(list_sdf_files(ligand_path))

    if ligand_batch is not None:
        if ud2lig_dir is not None:
            raise ValueError("A UD2LIG directory cannot be combined with -lb / ligand_batch.")
        sdf_file_names.extend(_read_ligand_batch_file(os.path.abspath(ligand_batch)))

    if ud2lig_dir is not None:
        return LIGAND_SOURCE_UD2LIG, tuple(), ud2lig_dir
    if sdf_file_names:
        return LIGAND_SOURCE_SDF_FILES, tuple(sdf_file_names), None
    raise ValueError("Ligand SDF file input not found!")


def resolve_docking_request(args, config=None):
    """Resolve docking inputs without invoking topology or GPU code."""
    if config is None:
        config = load_config_from_args(args)
    config = merge_cli_overrides(args, config, "docking")

    receptor = config.required.receptor
    if receptor is None:
        raise ValueError("Receptor file name not specified!")

    ligand_source, ligand_sdf_file_name_list, ud2lig_dir = resolve_ligand_inputs(
        config.required.ligand,
        config.required.ligand_batch,
        allow_ud2lig=True,
    )
    if ud2lig_dir is not None:
        manifest = load_ud2lig_manifest(os.path.join(ud2lig_dir, "manifest.json"))
        validate_ud2lig_against_config(manifest, config)

    root_temp_dir_name = os.path.abspath(config.preprocessing.temp_dir_name)
    return ResolvedDockingRequest(
        receptor_file_name=os.path.abspath(receptor),
        ligand_source=ligand_source,
        ligand_sdf_file_name_list=ligand_sdf_file_name_list,
        ud2lig_dir=ud2lig_dir,
        target_center=tuple(config.required.center),
        root_temp_dir_name=root_temp_dir_name,
        docking_pose_sdf_file_name=os.path.abspath(config.preprocessing.output_sdf),
        remove_temp_dir=root_temp_dir_name == "/tmp",
        config=config,
    )


def resolve_prepare_protein_request(args, config=None):
    """Resolve protein-preparation inputs without doing preparation work."""
    if config is None:
        config = load_config_from_args(args)
    config = merge_cli_overrides(args, config, "prepare_protein")

    receptor = config.required.receptor
    if receptor is None:
        raise ValueError("Receptor file name not specified!")

    output_dms = getattr(args, "output_dms", None)
    if not output_dms:
        raise ValueError("Output receptor DMS file (-o) is required!")

    root_temp_dir_name = os.path.abspath(config.preprocessing.temp_dir_name)
    return ResolvedPrepareProteinRequest(
        receptor_file_name=os.path.abspath(receptor),
        root_temp_dir_name=root_temp_dir_name,
        receptor_dms_file_name=os.path.abspath(output_dms),
        remove_temp_dir=root_temp_dir_name == "/tmp",
        config=config,
    )


def resolve_prepare_ligands_request(args, config=None):
    """Resolve ligand-preparation inputs without doing preparation work."""
    if config is None:
        config = load_config_from_args(args)
    config = merge_cli_overrides(args, config, "prepare_ligands")

    output_ud2lig_dir = getattr(args, "output_ud2lig_dir", None)
    if not output_ud2lig_dir:
        raise ValueError("Output UD2LIG directory (-o) is required!")

    _, ligand_sdf_file_name_list, _ = resolve_ligand_inputs(
        config.required.ligand,
        config.required.ligand_batch,
        allow_ud2lig=False,
    )

    root_temp_dir_name = os.path.abspath(config.preprocessing.temp_dir_name)
    return ResolvedPrepareLigandsRequest(
        ligand_sdf_file_name_list=ligand_sdf_file_name_list,
        output_ud2lig_dir=os.path.abspath(output_ud2lig_dir),
        root_temp_dir_name=root_temp_dir_name,
        remove_temp_dir=root_temp_dir_name == "/tmp",
        config=config,
    )
