from pathlib import Path
import warnings

import msys
import pytest

from unidock2.unidocktools.protein_topology import (
    MissingNonbondedTermsWarning,
    _read_receptor_force_field_data,
)
from unidock2.unidocktools.unidock_receptor_topology_builder import (
    UnidockReceptorTopologyBuilder,
)

TEST_RECEPTOR_DMS = Path(__file__).parents[2] / "data" / "receptor_topology" / "test_receptor_topology_protocol.dms"


class _FakeAtom:
    def __init__(self, charge):
        self.charge = charge


class _FakeNonbondedTable:
    def __init__(self, atom_types):
        self._atom_types = atom_types
        self.nterms = len(atom_types)

    def term(self, atom_idx):
        return {"type": self._atom_types[atom_idx]}


class _FakeSystem:
    def __init__(self, charges, nonbonded_table):
        self._atoms = [_FakeAtom(charge) for charge in charges]
        self._nonbonded_table = nonbonded_table
        self.natoms = len(self._atoms)

    def atom(self, atom_idx):
        return self._atoms[atom_idx]

    def getTable(self, table_name):
        assert table_name == "nonbonded"
        return self._nonbonded_table


@pytest.mark.parametrize("nonbonded_table", [None, _FakeNonbondedTable([])])
def test_missing_nonbonded_terms_use_fallback_types_and_safe_atom_charges(nonbonded_table):
    system = _FakeSystem([0.25, float("nan"), float("inf"), None], nonbonded_table)

    with pytest.warns(MissingNonbondedTermsWarning, match="no nonbonded terms"):
        ff_atom_types, charges = _read_receptor_force_field_data(system)

    assert ff_atom_types == ["c", "c", "c", "c"]
    assert charges == [0.25, 0.0, 0.0, 0.0]


def test_complete_nonbonded_terms_preserve_existing_types_and_charges():
    system = _FakeSystem(
        [-0.1, 0.2],
        _FakeNonbondedTable(["ca", "n"]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingNonbondedTermsWarning)
        ff_atom_types, charges = _read_receptor_force_field_data(system)

    assert ff_atom_types == ["ca", "n"]
    assert charges == [-0.1, 0.2]


def test_partially_parameterized_nonbonded_table_remains_an_error():
    system = _FakeSystem(
        [-0.1, 0.2],
        _FakeNonbondedTable(["ca"]),
    )

    with pytest.raises(ValueError, match="nonbonded term count"):
        _read_receptor_force_field_data(system)


def test_structure_only_dms_runs_the_complete_receptor_topology_path(tmp_path):
    parameterized_system = msys.LoadDMS(str(TEST_RECEPTOR_DMS))
    receptor_system = parameterized_system.clone(
        parameterized_system.residue(1).atoms,
        structure_only=True,
    )
    expected_charges = [atom.charge for atom in receptor_system.atoms]
    receptor_file = tmp_path / "structure_only_receptor.dms"
    msys.SaveDMS(receptor_system, str(receptor_file))

    assert receptor_system.getTable("nonbonded") is None
    receptor_builder = UnidockReceptorTopologyBuilder(
        str(receptor_file),
        prepared_hydrogen=True,
        working_dir_name=str(tmp_path),
    )
    receptor_builder.generate_receptor_topology()
    with pytest.warns(MissingNonbondedTermsWarning):
        receptor_builder.analyze_receptor_topology()

    atom_records = receptor_builder.atom_info_nested_list
    assert len(atom_records) == receptor_system.natoms
    assert all(len(atom_record) == 6 for atom_record in atom_records)
    assert [atom_record[4] for atom_record in atom_records] == [0] * receptor_system.natoms
    assert [atom_record[5] for atom_record in atom_records] == pytest.approx(expected_charges)
