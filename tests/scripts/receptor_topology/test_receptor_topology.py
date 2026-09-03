import math
import os

import pytest

from context import TEST_DATA_DIR

from unidock2.unidocktools.unidock_receptor_topology_builder import (
    UnidockReceptorTopologyBuilder,
)

TEST_RECEPTOR_DATA_DIR = os.path.join(TEST_DATA_DIR, 'receptor_topology')


def _build_receptor_topology(receptor_file_name, working_dir_name, prepared_hydrogen):
    receptor_builder = UnidockReceptorTopologyBuilder(
        receptor_file_name,
        prepared_hydrogen=prepared_hydrogen,
        covalent_residue_atom_info_list=None,
        working_dir_name=working_dir_name,
    )

    receptor_builder.generate_receptor_topology()
    receptor_builder.analyze_receptor_topology()
    receptor_builder.get_summary_receptor_info()

    return receptor_builder.atom_info_nested_list


def _assert_engine_ready_atom_records(atom_info_nested_list):
    """Every record must match the [x, y, z, vina_type, ff_type, charge] contract."""
    assert atom_info_nested_list

    for atom_idx, atom_record in enumerate(atom_info_nested_list):
        assert len(atom_record) == 6, f'atom {atom_idx} has {len(atom_record)} fields'
        x, y, z, vina_type, ff_type, charge = atom_record

        for value in (x, y, z, charge):
            assert isinstance(value, float) and math.isfinite(value)

        assert isinstance(vina_type, int) and 0 <= vina_type <= 20
        assert isinstance(ff_type, int) and ff_type >= 0


@pytest.mark.parametrize(
    'receptor_file_name,prepared_hydrogen',
    [
        ('test_receptor_topology_protocol.pdb', True),
        ('test_receptor_topology_protocol.dms', True),
        ('test_receptor_topology_RNA.pdb', False),
    ],
)
def test_receptor_topology_produces_engine_ready_atom_records(
    receptor_file_name,
    prepared_hydrogen,
    tmp_path,
):
    atom_info_nested_list = _build_receptor_topology(
        os.path.join(TEST_RECEPTOR_DATA_DIR, receptor_file_name),
        str(tmp_path),
        prepared_hydrogen,
    )

    _assert_engine_ready_atom_records(atom_info_nested_list)
