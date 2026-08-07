import argparse
import os

import pytest

from unidock2.cli._resolve import (
    resolve_docking_request,
    resolve_protein_prep_request,
)
from unidock2.cli.docking import CLICommand as DockingCLICommand
from unidock2.cli.protein_prep import CLICommand as ProteinPrepCLICommand
from unidock2.config import UnidockConfig


def _parser(command):
    parser = argparse.ArgumentParser()
    command.add_arguments(parser)
    return parser


def _parse(command, arguments):
    return _parser(command).parse_args(arguments)


def test_argparse_uses_none_only_as_the_source_sentinel():
    docking_args = _parse(DockingCLICommand, [])
    protein_prep_args = _parse(ProteinPrepCLICommand, [])

    assert vars(docking_args) == {
        "receptor": None,
        "ligand": None,
        "ligand_batch": None,
        "center": None,
        "exhaustiveness": None,
        "seed": None,
        "gpu_device_id": None,
        "box_size": None,
        "search_mode": None,
        "output_docking_pose_sdf_file_name": None,
        "configurations": None,
    }
    assert vars(protein_prep_args) == {
        "receptor": None,
        "output_receptor_dms_file_name": None,
        "configurations": None,
    }


def test_help_uses_business_defaults_from_the_pydantic_schema():
    help_text = " ".join(_parser(DockingCLICommand).format_help().split())

    assert "Docking box center coordinates (default: [0.0, 0.0, 0.0])" in help_text
    assert "Number of independent search tasks (default: 512)" in help_text
    assert "Native engine search mode (default: 'balance')" in help_text


def test_common_cli_values_override_yaml_even_when_equal_to_defaults(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    yaml_config = UnidockConfig.from_dict(
        {
            "Required": {
                "receptor": "yaml-receptor.pdb",
                "ligand": "yaml-ligand.sdf",
                "center": [9, 8, 7],
            },
            "Advanced": {"exhaustiveness": 256, "seed": 7},
            "Hardware": {"gpu_device_id": 0},
            "Settings": {
                "box_size": [40, 40, 40],
                "search_mode": "detail",
            },
            "Preprocessing": {"output_docking_pose_sdf_file_name": "yaml-output.sdf"},
        }
    )
    args = _parse(
        DockingCLICommand,
        [
            "-r",
            "cli-receptor.pdb",
            "-l",
            "cli-ligand.sdf",
            "-c",
            "0",
            "0",
            "0",
            "--box_size",
            "30",
            "30",
            "30",
            "-e",
            "512",
            "--seed",
            "1234567",
            "--gpu_device_id",
            "2",
            "--search_mode",
            "free",
            "-o",
            "unidock2_pose.sdf",
        ],
    )

    request = resolve_docking_request(args, yaml_config)

    assert request.receptor_file_name == str(tmp_path / "cli-receptor.pdb")
    assert request.ligand_sdf_file_name_list == (str(tmp_path / "cli-ligand.sdf"),)
    assert request.target_center == (0.0, 0.0, 0.0)
    assert request.docking_pose_sdf_file_name == str(tmp_path / "unidock2_pose.sdf")
    assert request.config.settings.box_size == [30.0, 30.0, 30.0]
    assert request.config.advanced.exhaustiveness == 512
    assert request.config.advanced.seed == 1234567
    assert request.config.hardware.gpu_device_id == 2
    assert request.config.settings.search_mode == "free"


def test_omitted_cli_values_preserve_yaml_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = UnidockConfig.from_dict(
        {
            "Required": {
                "receptor": "receptor.pdb",
                "ligand": "ligand.sdf",
                "center": [1, 2, 3],
            },
            "Advanced": {"exhaustiveness": 128},
            "Preprocessing": {"output_docking_pose_sdf_file_name": "yaml-output.sdf"},
        }
    )

    request = resolve_docking_request(_parse(DockingCLICommand, []), config)

    assert request.target_center == (1.0, 2.0, 3.0)
    assert request.config.advanced.exhaustiveness == 128
    assert request.docking_pose_sdf_file_name == str(tmp_path / "yaml-output.sdf")


def test_config_file_and_cli_are_connected_by_the_public_resolver(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """Required:
  receptor: yaml-receptor.pdb
  ligand: yaml-ligand.sdf
Advanced:
  exhaustiveness: 128
""",
        encoding="utf-8",
    )
    args = _parse(
        DockingCLICommand,
        ["-cf", str(config_file), "-e", "1024"],
    )

    request = resolve_docking_request(args)

    assert request.receptor_file_name == str(tmp_path / "yaml-receptor.pdb")
    assert request.ligand_sdf_file_name_list == (str(tmp_path / "yaml-ligand.sdf"),)
    assert request.config.advanced.exhaustiveness == 1024


def test_ligand_batch_and_single_ligand_keep_the_existing_order(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    batch_file = tmp_path / "ligands.txt"
    batch_file.write_text("batch-a.sdf\n\nsubdir/batch-b.sdf\n", encoding="utf-8")
    config = UnidockConfig.from_dict(
        {
            "Required": {
                "receptor": "receptor.pdb",
                "ligand": "single.sdf",
                "ligand_batch": str(batch_file),
            }
        }
    )

    request = resolve_docking_request(_parse(DockingCLICommand, []), config)

    assert request.ligand_sdf_file_name_list == (
        str(tmp_path / "single.sdf"),
        str(tmp_path / "batch-a.sdf"),
        str(tmp_path / "subdir" / "batch-b.sdf"),
    )


def test_protein_prep_uses_the_same_precedence_rules(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = UnidockConfig.from_dict(
        {
            "Required": {"receptor": "yaml-receptor.pdb"},
            "Preprocessing": {
                "output_receptor_dms_file_name": "yaml-receptor.dms",
                "temp_dir_name": str(tmp_path),
            },
        }
    )
    args = _parse(
        ProteinPrepCLICommand,
        ["-r", "cli-receptor.pdb", "-o", "receptor_parameterized.dms"],
    )

    request = resolve_protein_prep_request(args, config)

    assert request.receptor_file_name == str(tmp_path / "cli-receptor.pdb")
    assert request.receptor_dms_file_name == str(tmp_path / "receptor_parameterized.dms")
    assert request.root_temp_dir_name == str(tmp_path)
    assert not request.remove_temp_dir


def test_missing_required_runtime_inputs_have_clear_errors():
    args = _parse(DockingCLICommand, [])

    with pytest.raises(ValueError, match="Receptor file name not specified"):
        resolve_docking_request(args, UnidockConfig())

    config = UnidockConfig.from_dict({"Required": {"receptor": os.devnull}})
    with pytest.raises(ValueError, match="Ligand SDF file input not found"):
        resolve_docking_request(args, config)
