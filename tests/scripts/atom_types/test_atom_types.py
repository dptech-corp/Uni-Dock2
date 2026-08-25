import pytest
from rdkit import Chem

from unidock2.atom_types.smarts_atom_typer import SmartsAtomTyper, SmartsRule
from unidock2.atom_types.vina import (
    VINA_ATOM_TYPE_DICT,
    VINA_ATOM_TYPE_PROPERTY,
    VINA_ATOM_TYPE_RULES,
)


def test_later_smarts_rules_override_earlier_rules():
    mol = Chem.MolFromSmiles("C=O")
    atom_typer = SmartsAtomTyper(
        rules=(
            SmartsRule("[*]", "generic"),
            SmartsRule("[#6]=[#8]", "carbonyl_carbon"),
        ),
        property_name="test_atom_type",
    )

    atom_typer.assign_atom_types(mol)

    assert [atom.GetProp("test_atom_type") for atom in mol.GetAtoms()] == [
        "carbonyl_carbon",
        "generic",
    ]


def test_invalid_smarts_is_rejected_during_initialization():
    with pytest.raises(ValueError, match="Invalid SMARTS"):
        SmartsAtomTyper(
            rules=(SmartsRule("[", "invalid"),),
            property_name="test_atom_type",
        )


def test_unassigned_atoms_are_rejected():
    mol = Chem.MolFromSmiles("CO")
    atom_typer = SmartsAtomTyper(
        rules=(SmartsRule("[#6]", "carbon"),),
        property_name="test_atom_type",
    )

    with pytest.raises(ValueError, match=r"atom indices: \[1\]"):
        atom_typer.assign_atom_types(mol)


def test_vina_rules_preserve_reserved_engine_types():
    assigned_atom_types = {rule.atom_type for rule in VINA_ATOM_TYPE_RULES}

    assert assigned_atom_types == set(VINA_ATOM_TYPE_DICT) - {"O_P", "O_D"}
