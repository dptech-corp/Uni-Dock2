import pytest
from rdkit import Chem

from unidock2.utils.molecule_processing import (
    get_mol_with_indices,
    get_mol_without_indices,
)


ATOM_PROPERTIES = ("source_idx", "label", "score")


def _annotated_molecule():
    molecule = Chem.MolFromSmiles("CCNCP")
    molecule.SetProp("source_name", "annotated molecule")
    for atom in molecule.GetAtoms():
        atom_idx = atom.GetIdx()
        atom.SetIntProp("source_idx", atom_idx)
        atom.SetProp("label", f"atom-{atom_idx}")
        atom.SetDoubleProp("score", atom_idx + 0.25)

    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_idx in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_idx, (atom_idx, atom_idx + 0.5, -atom_idx))
    molecule.AddConformer(conformer)
    return molecule


def _atom_signature(molecule):
    return [
        (
            atom.GetSymbol(),
            atom.GetChiralTag(),
            atom.GetFormalCharge(),
            atom.GetNumExplicitHs(),
        )
        for atom in molecule.GetAtoms()
    ]


def _bond_signature(molecule):
    return [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) for bond in molecule.GetBonds()]


def test_selected_and_complement_copies_match_and_keep_source_atom_order():
    molecule = _annotated_molecule()
    selected_indices = [4, 2, 1]
    remove_indices = [0, 3]

    selected_molecule = get_mol_with_indices(
        molecule,
        selected_indices=selected_indices,
        keep_properties=ATOM_PROPERTIES,
        keep_mol_properties=("source_name",),
    )
    complement_molecule = get_mol_without_indices(
        molecule,
        remove_indices=remove_indices,
        keep_properties=ATOM_PROPERTIES,
        keep_mol_properties=("source_name",),
    )

    assert [atom.GetIntProp("source_idx") for atom in selected_molecule.GetAtoms()] == [1, 2, 4]
    assert _atom_signature(selected_molecule) == _atom_signature(complement_molecule)
    assert _bond_signature(selected_molecule) == _bond_signature(complement_molecule)
    assert selected_molecule.GetProp("source_name") == "annotated molecule"
    assert complement_molecule.GetProp("source_name") == "annotated molecule"

    for molecule_copy in (selected_molecule, complement_molecule):
        for atom in molecule_copy.GetAtoms():
            source_idx = atom.GetIntProp("source_idx")
            assert atom.GetProp("label") == f"atom-{source_idx}"
            assert atom.GetDoubleProp("score") == pytest.approx(source_idx + 0.25)
        assert molecule_copy.GetNumConformers() == 0


def test_dummy_atom_map_and_requested_r_group_properties_are_preserved():
    editable_molecule = Chem.RWMol()
    dummy_atom = Chem.Atom("*")
    dummy_atom.SetAtomMapNum(7)
    dummy_atom.SetProp("dummyLabel", "R7")
    dummy_atom.SetProp("_MolFileRLabel", "7")
    dummy_idx = editable_molecule.AddAtom(dummy_atom)
    carbon_idx = editable_molecule.AddAtom(Chem.Atom("C"))
    editable_molecule.AddBond(dummy_idx, carbon_idx, Chem.BondType.SINGLE)
    molecule = editable_molecule.GetMol()

    copied_molecule = get_mol_with_indices(
        molecule,
        selected_indices=[dummy_idx],
        keep_properties=("dummyLabel", "_MolFileRLabel"),
    )

    copied_atom = copied_molecule.GetAtomWithIdx(0)
    assert copied_atom.GetAtomicNum() == 0
    assert copied_atom.GetSymbol() == "R7"
    assert copied_atom.GetAtomMapNum() == 7
    assert copied_atom.GetProp("dummyLabel") == "R7"
    assert copied_atom.GetProp("_MolFileRLabel") == "7"


@pytest.mark.parametrize(
    ("smiles", "expected_explicit_hydrogen_delta"),
    [
        ("CN", 1),
        ("CP", 1),
        ("CC", 0),
    ],
)
def test_cut_bond_adds_explicit_hydrogen_only_to_retained_nitrogen_or_phosphorus(
    smiles,
    expected_explicit_hydrogen_delta,
):
    molecule = Chem.MolFromSmiles(smiles)
    original_explicit_hydrogens = molecule.GetAtomWithIdx(1).GetNumExplicitHs()

    selected_molecule = get_mol_with_indices(molecule, selected_indices=[1])
    complement_molecule = get_mol_without_indices(molecule, remove_indices=[0])

    expected_explicit_hydrogens = original_explicit_hydrogens + expected_explicit_hydrogen_delta
    assert selected_molecule.GetAtomWithIdx(0).GetNumExplicitHs() == expected_explicit_hydrogens
    assert complement_molecule.GetAtomWithIdx(0).GetNumExplicitHs() == expected_explicit_hydrogens


def test_atom_chemistry_and_bond_types_are_preserved_when_all_atoms_are_selected():
    molecule = Chem.MolFromSmiles("[NH3+][C@H](F)C=C")

    copied_molecule = get_mol_with_indices(
        molecule,
        selected_indices=list(reversed(range(molecule.GetNumAtoms()))),
    )

    assert _atom_signature(copied_molecule) == _atom_signature(molecule)
    assert _bond_signature(copied_molecule) == _bond_signature(molecule)
