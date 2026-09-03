"""Invariants the per-residue receptor split has to hold.

Engine records only read atom properties, so they stay correct even when the
bonds inside a residue are lost. Covalent docking does walk the bonds, which is
why connectivity is asserted separately here.
"""

from pathlib import Path

import msys
import pytest

from unidock2.unidocktools.protein_topology import prepare_receptor_residue_mol_list

TEST_RECEPTOR_DMS = (
    Path(__file__).parents[2] / "data" / "receptor_topology" / "test_receptor_topology_protocol.dms"
)


@pytest.fixture(scope="module")
def receptor_residue_mols():
    system = msys.LoadDMS(str(TEST_RECEPTOR_DMS))
    protein_mol, protein_residue_mols, _cofactor_residue_mols = (
        prepare_receptor_residue_mol_list(system)
    )
    return protein_mol, protein_residue_mols


def _atom_residue_indices(mol):
    return [atom.GetIntProp("internal_residue_idx") for atom in mol.GetAtoms()]


def test_residues_partition_every_protein_atom(receptor_residue_mols):
    protein_mol, protein_residue_mols = receptor_residue_mols

    assert sum(mol.GetNumAtoms() for mol in protein_residue_mols) == protein_mol.GetNumAtoms()

    residue_indices = []
    for residue_mol in protein_residue_mols:
        indices = set(_atom_residue_indices(residue_mol))
        assert len(indices) == 1, "one residue mol must not mix several residues"
        residue_indices.append(indices.pop())

    assert len(set(residue_indices)) == len(residue_indices)
    assert residue_indices == sorted(residue_indices)


def test_residues_keep_source_atom_order(receptor_residue_mols):
    protein_mol, protein_residue_mols = receptor_residue_mols

    expected = [atom.GetIntProp("internal_atom_idx") for atom in protein_mol.GetAtoms()]
    actual = [
        atom.GetIntProp("internal_atom_idx")
        for residue_mol in protein_residue_mols
        for atom in residue_mol.GetAtoms()
    ]

    assert actual == sorted(actual)
    assert actual == expected


def test_bonds_inside_a_residue_survive_and_bonds_between_residues_do_not(
    receptor_residue_mols,
):
    protein_mol, protein_residue_mols = receptor_residue_mols

    atom_residue = _atom_residue_indices(protein_mol)
    expected_intra_residue_bonds = sum(
        1
        for bond in protein_mol.GetBonds()
        if atom_residue[bond.GetBeginAtomIdx()] == atom_residue[bond.GetEndAtomIdx()]
    )

    assert expected_intra_residue_bonds > 0
    assert sum(mol.GetNumBonds() for mol in protein_residue_mols) == expected_intra_residue_bonds


def test_every_hydrogen_stays_bonded_to_its_heavy_atom(receptor_residue_mols):
    """Guards find_covalent_hydrogen_atoms(), which walks GetNeighbors()."""
    _protein_mol, protein_residue_mols = receptor_residue_mols

    hydrogen_count = 0
    for residue_mol in protein_residue_mols:
        for atom in residue_mol.GetAtoms():
            if atom.GetSymbol() != "H":
                continue
            hydrogen_count += 1
            neighbours = atom.GetNeighbors()
            assert len(neighbours) == 1, "a hydrogen must hang off exactly one atom"
            assert neighbours[0].GetSymbol() != "H"

    assert hydrogen_count > 0
