import os

import pytest

from unidock2._engine import (
    DEFAULT_ENGINE_OUTPUT_PREFIX,
    build_engine_request,
)
from unidock2.config import UnidockConfig
from unidock2.unidocktools.unidock_protocol_runner import UnidockProtocolRunner
from unidock2.unidocktools.unidock_protocol_runner import (
    _ud2lig_dir_for_pose_sdf,
    _write_ud2lig_checkpoint,
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


def test_from_config_builds_the_complete_native_request(tmp_path):
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
        bias="align",
        bias_k=0.75,
        use_tor_lib=True,
        energy_decomp=True,
        template_docking=True,
        compute_center=False,
        gpu_device_id=2,
        max_gpu_memory=2048,
    )
    runner = UnidockProtocolRunner.from_config(
        "receptor.pdb",
        ["ligand.sdf"],
        (1, 2, 3),
        config=config,
        working_dir_name=str(tmp_path),
        docking_pose_sdf_file_name=str(tmp_path / "poses.sdf"),
    )

    assert build_engine_request(
        runner._current_config(),
        target_center=runner.target_center,
        output_dir=runner.unidock2_output_dir_name,
        receptor=[{"atom": "receptor"}],
        ligands={"ligand_0": {"atom": "ligand"}},
    ) == {
        "parameters": {
            "center": [1.0, 2.0, 3.0],
            "box_size": [10.0, 20.0, 30.0],
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
            "bias": "align",
            "bias_k": 0.75,
            "use_tor_lib": True,
            "energy_decomp": True,
            "constraint_docking": True,
        },
        "runtime": {
            "output_dir": str(tmp_path / "unidock2_output"),
            "output_prefix": DEFAULT_ENGINE_OUTPUT_PREFIX,
            "gpu_device_id": 2,
            "max_gpu_memory": 2048,
        },
        "molecules": {
            "receptor": [{"atom": "receptor"}],
            "ligand_0": {"atom": "ligand"},
        },
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


def test_mutating_legacy_public_attributes_still_affects_engine_request(tmp_path):
    runner = UnidockProtocolRunner(
        "receptor.pdb",
        ["ligand.sdf"],
        (1, 2, 3),
        working_dir_name=str(tmp_path),
    )
    runner.mc_steps = 99
    runner.box_size = [7, 8, 9]

    request = build_engine_request(
        runner._current_config(),
        target_center=runner.target_center,
        output_dir=runner.unidock2_output_dir_name,
        receptor=[],
        ligands={},
    )

    assert request["parameters"]["mc_steps"] == 99
    assert request["parameters"]["box_size"] == [7.0, 8.0, 9.0]
    assert os.path.isdir(runner.unidock2_output_dir_name)


def test_engine_checkpoint_writes_ud2lig_next_to_pose_sdf(tmp_path):
    from rdkit import Chem

    from context import TEST_DATA_DIR
    from unidock2.io.ud2lig import read_ud2lig

    ligand_file = os.path.join(TEST_DATA_DIR, "ligand_topology", "test_vina_atom_type_1.sdf")
    molecule = Chem.SDMolSupplier(ligand_file, removeHs=False)[0]
    molecule.SetProp("ud2_molecule_name", "MOL_0")
    conformer = molecule.GetConformer()
    atoms = []
    for atom_idx in range(molecule.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_idx)
        atoms.append([position.x, position.y, position.z, 2, 0, 0.0, [], []])
    topology = {
        "atoms": atoms,
        "torsions": [],
        "root_atoms": [0],
        "fragment_atom_idx": [[0]],
    }

    pose_sdf = tmp_path / "poses.sdf"
    output_dir = _ud2lig_dir_for_pose_sdf(str(pose_sdf))
    assert output_dir.endswith("poses.ud2lig")

    _write_ud2lig_checkpoint(
        output_dir,
        {"MOL_0": topology},
        [molecule],
        UnidockConfig(),
    )

    loaded_info, loaded_mols, manifest = read_ud2lig(output_dir)
    assert manifest["n_ligands"] == 1
    assert list(loaded_info) == ["MOL_0"]
    assert loaded_mols[0].GetProp("ud2_molecule_name") == "MOL_0"
