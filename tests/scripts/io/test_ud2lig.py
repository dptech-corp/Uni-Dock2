import os

import pytest
from rdkit import Chem

from context import TEST_DATA_DIR
from unidock2.config import UnidockConfig
from unidock2.io.ud2lig import (
    UD2LIG_MAGIC,
    prep_from_config,
    read_ud2lig,
    validate_ud2lig_against_config,
    write_ud2lig,
)


def _named_mol(name):
    ligand_file = os.path.join(TEST_DATA_DIR, "ligand_topology", "test_vina_atom_type_1.sdf")
    molecule = Chem.SDMolSupplier(ligand_file, removeHs=False)[0]
    molecule.SetProp("ud2_molecule_name", name)
    return molecule


def _topology(molecule):
    conformer = molecule.GetConformer()
    atoms = []
    for atom_idx in range(molecule.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_idx)
        atoms.append([position.x, position.y, position.z, 2, 0, 0.0, [], []])
    return {
        "atoms": atoms,
        "torsions": [],
        "root_atoms": [0],
        "fragment_atom_idx": [[0]],
    }


def test_write_and_read_preserves_topology_and_atom_order(tmp_path):
    mol_0 = _named_mol("MOL_0")
    mol_1 = _named_mol("MOL_1")
    ligand_info = {"MOL_0": _topology(mol_0), "MOL_1": _topology(mol_1)}
    output_dir = tmp_path / "lib.ud2lig"

    write_ud2lig(
        output_dir,
        ligand_info,
        [mol_0, mol_1],
        prep_from_config(UnidockConfig()),
        shard_size=1,
    )

    loaded_info, loaded_mols, manifest = read_ud2lig(output_dir)

    assert manifest["magic"] == UD2LIG_MAGIC
    assert manifest["n_ligands"] == 2
    assert len(manifest["shards"]) == 2
    assert list(loaded_info) == ["MOL_0", "MOL_1"]
    assert [mol.GetProp("ud2_molecule_name") for mol in loaded_mols] == ["MOL_0", "MOL_1"]
    assert loaded_mols[0].GetNumAtoms() == len(loaded_info["MOL_0"]["atoms"])
    assert loaded_info["MOL_0"]["root_atoms"] == [0]


def test_write_ud2lig_overwrite_replaces_existing_library(tmp_path):
    mol_0 = _named_mol("MOL_0")
    output_dir = tmp_path / "lib.ud2lig"
    write_ud2lig(
        output_dir,
        {"MOL_0": _topology(mol_0)},
        [mol_0],
        prep_from_config(UnidockConfig()),
    )
    with pytest.raises(ValueError, match="not empty"):
        write_ud2lig(
            output_dir,
            {"MOL_0": _topology(mol_0)},
            [mol_0],
            prep_from_config(UnidockConfig()),
        )

    write_ud2lig(
        output_dir,
        {"MOL_0": _topology(mol_0)},
        [mol_0],
        prep_from_config(UnidockConfig()),
        overwrite=True,
    )
    _, loaded_mols, manifest = read_ud2lig(output_dir)
    assert manifest["n_ligands"] == 1
    assert loaded_mols[0].GetProp("ud2_molecule_name") == "MOL_0"


def test_validate_ud2lig_rejects_construct_ff_mismatch(tmp_path):
    mol_0 = _named_mol("MOL_0")
    output_dir = tmp_path / "lib.ud2lig"
    write_ud2lig(
        output_dir,
        {"MOL_0": _topology(mol_0)},
        [mol_0],
        prep_from_config(UnidockConfig()),
    )
    _, _, manifest = read_ud2lig(output_dir)

    with pytest.raises(ValueError, match="construct_ff"):
        validate_ud2lig_against_config(manifest, UnidockConfig().with_overrides(construct_ff=True))
