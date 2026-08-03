import json
import os

import pytest
from rdkit import Chem

from context import TEST_DATA_DIR

from unidock2.unidocktools.unidock_ligand_pose_writer import (
    UnidockLigandPoseWriter,
)
from unidock2.unidocktools.unidock_ligand_topology_builder import (
    UnidockLigandTopologyBuilder,
)


ENERGY_PROPERTIES = (
    "vina_binding_free_energy",
    "vina_intra_inter",
    "vina_intra",
    "vina_inter",
    "vina_box_penalty",
    "vina_torsion_number_energy",
)
NUM_SCREENING_LIGANDS = 224
NUM_FAKE_BATCHES = 7
TWO_POSE_LIGAND_IDX = 37


@pytest.fixture(scope="module")
def screening_ligands():
    ligand_file_name = os.path.join(
        TEST_DATA_DIR,
        "free_docking",
        "virtual_screening",
        "actives_cleaned.sdf",
    )
    builder = UnidockLigandTopologyBuilder(
        [ligand_file_name],
        n_cpu=1,
    )

    assert builder.num_ligands == NUM_SCREENING_LIGANDS
    for ligand_idx, ligand_mol in enumerate(builder.ligand_mol_list):
        assert ligand_mol.GetProp("ud2_molecule_name") == f"MOL_{ligand_idx}"
        assert ligand_mol.GetIntProp("source_mol_idx") == ligand_idx
        assert ligand_mol.GetProp("source_sdf_file_name") == ligand_file_name

    return builder.ligand_mol_list


def _pose_coordinates(ligand_mol, ligand_idx, pose_idx):
    conformer = ligand_mol.GetConformer()
    offset = ligand_idx * 0.1 + pose_idx * 0.01
    coordinates = []
    for atom_idx in range(ligand_mol.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_idx)
        coordinates.extend(
            [
                position.x + offset,
                position.y - offset,
                position.z + 2 * offset,
            ]
        )
    return coordinates


def _pose_energy(ligand_idx, pose_idx):
    if ligand_idx == TWO_POSE_LIGAND_IDX:
        binding_energy = -5.0 if pose_idx == 0 else -9.0
    else:
        binding_energy = -(ligand_idx + 1.0) - pose_idx * 0.25

    return [
        binding_energy,
        1000.0 + ligand_idx + pose_idx * 0.1,
        2000.0 + ligand_idx + pose_idx * 0.1,
        3000.0 + ligand_idx + pose_idx * 0.1,
        4000.0 + ligand_idx + pose_idx * 0.1,
        5000.0 + ligand_idx + pose_idx * 0.1,
        6000.0 + ligand_idx + pose_idx * 0.1,
    ]


def _make_pose(ligand_mol, ligand_idx, pose_idx):
    return {
        "coords": _pose_coordinates(ligand_mol, ligand_idx, pose_idx),
        "energy": _pose_energy(ligand_idx, pose_idx),
        "decomp": {
            "ligand_idx": ligand_idx,
            "pose_idx": pose_idx,
        },
    }


def _write_shuffled_pose_batches(tmp_path, ligand_mol_list):
    batches = [{} for _ in range(NUM_FAKE_BATCHES)]
    expected_poses = []

    for ligand_idx, ligand_mol in enumerate(ligand_mol_list):
        ligand_name = ligand_mol.GetProp("ud2_molecule_name")
        num_poses = 2 if ligand_idx == TWO_POSE_LIGAND_IDX else 1
        poses = []
        for pose_idx in range(num_poses):
            pose = _make_pose(ligand_mol, ligand_idx, pose_idx)
            poses.append(pose)
            expected_poses.append(
                {
                    "ligand_idx": ligand_idx,
                    "pose_idx": pose_idx,
                    "pose": pose,
                }
            )
        batches[ligand_idx % NUM_FAKE_BATCHES][ligand_name] = poses

    batch_file_names = []
    for batch_idx, batch in enumerate(batches):
        batch_file_name = tmp_path / f"batch_{batch_idx}.json"
        reversed_batch = dict(reversed(list(batch.items())))
        batch_file_name.write_text(json.dumps(reversed_batch))
        batch_file_names.append(str(batch_file_name))

    return list(reversed(batch_file_names)), expected_poses


def _read_valid_sdf_molecules(sdf_file_name):
    molecules = list(Chem.SDMolSupplier(str(sdf_file_name), removeHs=False))
    assert all(molecule is not None for molecule in molecules)
    return molecules


def test_pose_writer_preserves_ligand_order_across_shuffled_batches(
    screening_ligands,
    tmp_path,
):
    batch_file_names, expected_poses = _write_shuffled_pose_batches(
        tmp_path,
        screening_ligands,
    )
    output_file_name = tmp_path / "screening_poses.sdf"

    pose_writer = UnidockLigandPoseWriter(
        screening_ligands,
        batch_file_names,
        energy_decomp=True,
        docking_pose_sdf_file_name=str(output_file_name),
    )
    pose_writer.generate_docking_pose_sdf()

    output_molecules = _read_valid_sdf_molecules(output_file_name)
    assert len(output_molecules) == NUM_SCREENING_LIGANDS + 1
    assert len(output_molecules) == len(expected_poses)

    for output_mol, expected in zip(output_molecules, expected_poses):
        ligand_idx = expected["ligand_idx"]
        pose_idx = expected["pose_idx"]
        pose = expected["pose"]
        source_mol = screening_ligands[ligand_idx]

        assert (
            output_mol.GetProp("ud2_molecule_name")
            == f"MOL_{ligand_idx}_unidock2_pose_{pose_idx}"
        )
        assert int(output_mol.GetProp("source_mol_idx")) == ligand_idx
        assert (
            output_mol.GetProp("source_sdf_file_name")
            == source_mol.GetProp("source_sdf_file_name")
        )
        assert (
            output_mol.GetProp("ligand_molecule_name")
            == source_mol.GetProp("ligand_molecule_name")
        )
        assert output_mol.GetNumAtoms() == source_mol.GetNumAtoms()

        output_conformer = output_mol.GetConformer()
        expected_coordinates = pose["coords"]
        for atom_idx in range(output_mol.GetNumAtoms()):
            output_position = output_conformer.GetAtomPosition(atom_idx)
            coord_idx = atom_idx * 3
            assert output_position.x == pytest.approx(
                expected_coordinates[coord_idx],
                abs=1e-3,
            )
            assert output_position.y == pytest.approx(
                expected_coordinates[coord_idx + 1],
                abs=1e-3,
            )
            assert output_position.z == pytest.approx(
                expected_coordinates[coord_idx + 2],
                abs=1e-3,
            )

        for energy_idx, property_name in enumerate(ENERGY_PROPERTIES):
            assert float(output_mol.GetProp(property_name)) == pytest.approx(
                pose["energy"][energy_idx]
            )
        assert json.loads(output_mol.GetProp("decomp")) == pose["decomp"]

    selected_pose_names = [
        molecule.GetProp("ud2_molecule_name")
        for molecule in output_molecules
        if int(molecule.GetProp("source_mol_idx")) == TWO_POSE_LIGAND_IDX
    ]
    assert selected_pose_names == [
        f"MOL_{TWO_POSE_LIGAND_IDX}_unidock2_pose_0",
        f"MOL_{TWO_POSE_LIGAND_IDX}_unidock2_pose_1",
    ]
    selected_energies = [
        float(molecule.GetProp("vina_binding_free_energy"))
        for molecule in output_molecules
        if int(molecule.GetProp("source_mol_idx")) == TWO_POSE_LIGAND_IDX
    ]
    assert selected_energies == [-5.0, -9.0]


def test_pose_writer_omits_decomp_when_disabled(screening_ligands, tmp_path):
    ligand_mol = screening_ligands[0]
    pose_file_name = tmp_path / "pose_with_decomp.json"
    pose_file_name.write_text(
        json.dumps(
            {
                "MOL_0": [_make_pose(ligand_mol, 0, 0)],
            }
        )
    )
    output_file_name = tmp_path / "pose_without_decomp.sdf"

    pose_writer = UnidockLigandPoseWriter(
        [ligand_mol],
        [str(pose_file_name)],
        energy_decomp=False,
        docking_pose_sdf_file_name=str(output_file_name),
    )
    pose_writer.generate_docking_pose_sdf()

    output_mol = _read_valid_sdf_molecules(output_file_name)[0]
    assert not output_mol.HasProp("decomp")


def test_pose_writer_rejects_incorrect_coordinate_count(screening_ligands, tmp_path):
    ligand_mol = screening_ligands[0]
    invalid_pose = _make_pose(ligand_mol, 0, 0)
    invalid_pose["coords"].pop()

    pose_file_name = tmp_path / "invalid_coordinates.json"
    pose_file_name.write_text(json.dumps({"MOL_0": [invalid_pose]}))
    output_file_name = tmp_path / "invalid_coordinates.sdf"
    pose_writer = UnidockLigandPoseWriter(
        [ligand_mol],
        [str(pose_file_name)],
        docking_pose_sdf_file_name=str(output_file_name),
    )

    try:
        with pytest.raises(ValueError, match="equal number of atoms"):
            pose_writer.generate_docking_pose_sdf()
    finally:
        if hasattr(pose_writer, "docking_pose_writer"):
            pose_writer.docking_pose_writer.close()


def test_pose_writer_rejects_missing_ligand_result(screening_ligands, tmp_path):
    first_ligand = screening_ligands[0]
    pose_file_name = tmp_path / "missing_ligand.json"
    pose_file_name.write_text(
        json.dumps(
            {
                "MOL_0": [_make_pose(first_ligand, 0, 0)],
            }
        )
    )
    output_file_name = tmp_path / "missing_ligand.sdf"
    pose_writer = UnidockLigandPoseWriter(
        screening_ligands[:2],
        [str(pose_file_name)],
        docking_pose_sdf_file_name=str(output_file_name),
    )

    try:
        with pytest.raises(KeyError, match="MOL_1"):
            pose_writer.generate_docking_pose_sdf()
    finally:
        if hasattr(pose_writer, "docking_pose_writer"):
            pose_writer.docking_pose_writer.close()
