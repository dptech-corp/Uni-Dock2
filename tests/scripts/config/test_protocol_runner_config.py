import os

import pytest

from unidock2.config import UnidockConfig
from unidock2.unidocktools.unidock_protocol_runner import (
    UnidockProtocolRunner,
    _build_pipeline_kwargs,
)


def test_legacy_runner_defaults_come_from_the_config_schema(tmp_path):
    runner = UnidockProtocolRunner(
        "receptor.pdb",
        ["ligand.sdf"],
        (1, 2, 3),
        working_dir_name=str(tmp_path),
    )
    defaults = UnidockConfig()

    for field_name, expected in defaults.to_protocol_kwargs().items():
        if field_name == "center":
            continue
        assert getattr(runner, field_name) == expected
    assert runner.target_center == (1.0, 2.0, 3.0)


def test_legacy_optional_positional_order_remains_supported(tmp_path):
    runner = UnidockProtocolRunner(
        "receptor.pdb",
        ["ligand.sdf"],
        (1, 2, 3),
        (4, 5, 6),
        None,
        True,
        "reference.sdf",
        False,
        [{"1": 2}],
        False,
        None,
        True,
        False,
        True,
        str(tmp_path),
        "poses.sdf",
    )

    assert runner.box_size == [4.0, 5.0, 6.0]
    assert runner.template_docking
    assert not runner.compute_center
    assert runner.core_atom_mapping_dict_list == [{1: 2}]
    assert runner.construct_ff
    assert runner.preserve_receptor_hydrogen
    assert runner.docking_pose_sdf_file_name == os.path.abspath("poses.sdf")


def test_legacy_runner_keeps_optional_none_core_mappings(tmp_path):
    runner = UnidockProtocolRunner(
        "receptor.pdb",
        ["ligand.sdf"],
        (1, 2, 3),
        core_atom_mapping_dict_list=[None],
        working_dir_name=str(tmp_path),
    )

    assert runner.core_atom_mapping_dict_list == [None]


def test_from_config_and_pipeline_kwargs_preserve_the_native_contract(tmp_path):
    config = UnidockConfig().with_overrides(
        box_size=[10, 20, 30],
        task="score",
        search_mode="free",
        exhaustiveness=64,
        randomize=False,
        mc_steps=12,
        opt_steps=13,
        refine_steps=14,
        num_pose=4,
        rmsd_limit=1.5,
        energy_range=6.0,
        seed=42,
        use_tor_lib=True,
        energy_decomp=True,
        template_docking=True,
        compute_center=False,
        gpu_device_id=2,
    )
    runner = UnidockProtocolRunner.from_config(
        "receptor.pdb",
        ["ligand.sdf"],
        (1, 2, 3),
        config=config,
        working_dir_name=str(tmp_path),
        docking_pose_sdf_file_name=str(tmp_path / "poses.sdf"),
    )

    assert _build_pipeline_kwargs(
        runner._current_config(),
        runner.target_center,
        runner.unidock2_output_dir_name,
    ) == {
        "output_dir": str(tmp_path / "unidock2_output"),
        "center_x": 1.0,
        "center_y": 2.0,
        "center_z": 3.0,
        "size_x": 10.0,
        "size_y": 20.0,
        "size_z": 30.0,
        "task": "score",
        "search_mode": "free",
        "exhaustiveness": 64,
        "randomize": False,
        "mc_steps": 12,
        "opt_steps": 13,
        "refine_steps": 14,
        "num_pose": 4,
        "rmsd_limit": 1.5,
        "energy_range": 6.0,
        "seed": 42,
        "use_tor_lib": True,
        "energy_decomp": True,
        "constraint_docking": True,
        "gpu_device_id": 2,
    }


def test_runner_rejects_unknown_config_overrides(tmp_path):
    with pytest.raises(TypeError, match="unknown_option"):
        UnidockProtocolRunner(
            "receptor.pdb",
            ["ligand.sdf"],
            (1, 2, 3),
            working_dir_name=str(tmp_path),
            unknown_option=True,
        )


def test_mutating_legacy_public_attributes_still_affects_pipeline_kwargs(tmp_path):
    runner = UnidockProtocolRunner(
        "receptor.pdb",
        ["ligand.sdf"],
        (1, 2, 3),
        working_dir_name=str(tmp_path),
    )
    runner.mc_steps = 99
    runner.box_size = [7, 8, 9]

    kwargs = _build_pipeline_kwargs(
        runner._current_config(),
        runner.target_center,
        runner.unidock2_output_dir_name,
    )

    assert kwargs["mc_steps"] == 99
    assert (kwargs["size_x"], kwargs["size_y"], kwargs["size_z"]) == (
        7.0,
        8.0,
        9.0,
    )
    assert os.path.isdir(runner.unidock2_output_dir_name)
