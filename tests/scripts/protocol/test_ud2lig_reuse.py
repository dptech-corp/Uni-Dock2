import os

import pytest
from rdkit import Chem

from context import TEST_DATA_DIR
from unidock2.config import UnidockConfig
from unidock2.io.get_temp_dir_prefix import get_temp_dir_prefix
from unidock2.io.tempfile import TemporaryDirectory
from unidock2.io.ud2lig import prep_from_config, write_ud2lig
from unidock2.unidocktools.unidock_ligand_topology_builder import (
    UnidockLigandTopologyBuilder,
)
from unidock2.unidocktools.unidock_protocol_runner import UnidockProtocolRunner


ENERGY_PROPERTIES = (
    "vina_binding_free_energy",
    "vina_intra_inter",
    "vina_intra",
    "vina_inter",
    "vina_box_penalty",
    "vina_torsion_number_energy",
)


def _pose_records(sdf_file_name):
    molecules = list(Chem.SDMolSupplier(sdf_file_name, removeHs=False))
    assert molecules
    assert all(molecule is not None for molecule in molecules)

    records = []
    for molecule in molecules:
        conformer = molecule.GetConformer()
        records.append(
            {
                "name": molecule.GetProp("ud2_molecule_name"),
                "symbols": [atom.GetSymbol() for atom in molecule.GetAtoms()],
                "coords": [
                    (
                        conformer.GetAtomPosition(atom_idx).x,
                        conformer.GetAtomPosition(atom_idx).y,
                        conformer.GetAtomPosition(atom_idx).z,
                    )
                    for atom_idx in range(molecule.GetNumAtoms())
                ],
                "energies": {
                    property_name: float(molecule.GetProp(property_name))
                    for property_name in ENERGY_PROPERTIES
                },
            }
        )
    return records


def _assert_pose_sdfs_align(left_sdf, right_sdf):
    left_records = _pose_records(left_sdf)
    right_records = _pose_records(right_sdf)
    assert [record["name"] for record in left_records] == [
        record["name"] for record in right_records
    ]

    for left, right in zip(left_records, right_records):
        assert left["symbols"] == right["symbols"]
        assert len(left["coords"]) == len(right["coords"])
        for left_coord, right_coord in zip(left["coords"], right["coords"]):
            assert left_coord == pytest.approx(right_coord, abs=1e-3)
        for property_name in ENERGY_PROPERTIES:
            assert left["energies"][property_name] == pytest.approx(
                right["energies"][property_name],
                abs=1e-4,
            )


def test_prepare_then_ud2lig_docking_matches_live_sdf_prep():
    receptor = os.path.join(
        TEST_DATA_DIR,
        "free_docking",
        "molecular_docking",
        "1G9V_protein_water_cleaned.pdb",
    )
    ligand = os.path.join(
        TEST_DATA_DIR,
        "free_docking",
        "molecular_docking",
        "ligand_prepared.sdf",
    )
    pocket_center = (5.122, 18.327, 37.332)
    docking_kwargs = {
        "box_size": (30.0, 30.0, 30.0),
        "search_mode": "free",
        "exhaustiveness": 16,
        "mc_steps": 20,
        "opt_steps": -1,
        "refine_steps": 1,
        "num_pose": 3,
        "seed": 1234567,
        "engine_checkpoint": False,
    }

    root_temp_dir_name = "/tmp"
    temp_dir_prefix = os.path.join(
        root_temp_dir_name, get_temp_dir_prefix("test_ud2lig_reuse")
    )

    with TemporaryDirectory(prefix=temp_dir_prefix, delete=True) as working_dir_name:
        ligand_builder = UnidockLigandTopologyBuilder(
            [ligand],
            n_cpu=1,
            working_dir_name=working_dir_name,
        )
        ligand_builder.generate_batch_ligand_topology()
        ligand_builder.get_summary_ligand_info_dict()
        library = os.path.join(working_dir_name, "mylibrary.ud2lig")
        write_ud2lig(
            library,
            ligand_builder.summary_ligand_info_dict,
            ligand_builder.ligand_mol_list,
            prep_from_config(UnidockConfig()),
        )

        live_sdf = os.path.join(working_dir_name, "from_sdf.sdf")
        UnidockProtocolRunner(
            receptor,
            [ligand],
            target_center=pocket_center,
            working_dir_name=os.path.join(working_dir_name, "live"),
            docking_pose_sdf_file_name=live_sdf,
            **docking_kwargs,
        ).run_unidock_protocol()

        reused_sdf = os.path.join(working_dir_name, "from_ud2lig.sdf")
        UnidockProtocolRunner(
            receptor,
            [],
            target_center=pocket_center,
            working_dir_name=os.path.join(working_dir_name, "reuse"),
            docking_pose_sdf_file_name=reused_sdf,
            ud2lig_dir=library,
            **docking_kwargs,
        ).run_unidock_protocol()

        _assert_pose_sdfs_align(live_sdf, reused_sdf)
