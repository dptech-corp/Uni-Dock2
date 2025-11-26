import os
import pytest

from context import TEST_DATA_DIR

from unidock_processing.io.get_temp_dir_prefix import get_temp_dir_prefix
from unidock_processing.io.tempfile import TemporaryDirectory
from unidock_processing.unidocktools.unidock_receptor_topology_builder import (
    UnidockReceptorTopologyBuilder,
)

TEST_RECEPTOR_DATA_DIR = os.path.join(TEST_DATA_DIR, 'receptor_topology')

@pytest.fixture
def receptor_topology_test_pdb_file():
    receptor_pdb_file_name = os.path.join(
        TEST_RECEPTOR_DATA_DIR, 'test_receptor_topology_protocol.pdb'
    )

    return receptor_pdb_file_name

@pytest.fixture
def receptor_topology_test_dms_file():
    receptor_dms_file_name = os.path.join(
        TEST_RECEPTOR_DATA_DIR, 'test_receptor_topology_protocol.dms'
    )

    return receptor_dms_file_name

@pytest.fixture
def receptor_topology_RNA_test_pdb_file():
    receptor_pdb_file_name = os.path.join(
        TEST_RECEPTOR_DATA_DIR, 'test_receptor_topology_RNA.pdb'
    )

    return receptor_pdb_file_name

def test_receptor_topology_pdb(receptor_topology_test_pdb_file):
    root_temp_dir_name = '/tmp'
    temp_dir_prefix = os.path.join(
        root_temp_dir_name, get_temp_dir_prefix('test_receptor_topology')
    )

    with TemporaryDirectory(prefix=temp_dir_prefix, delete=True) as working_dir_name:
        unidock_receptor_topology_builder = UnidockReceptorTopologyBuilder(
            receptor_topology_test_pdb_file,
            prepared_hydrogen=True,
            covalent_residue_atom_info_list=None,
            working_dir_name=working_dir_name,
        )

        unidock_receptor_topology_builder.generate_receptor_topology()
        unidock_receptor_topology_builder.analyze_receptor_topology()
        unidock_receptor_topology_builder.get_summary_receptor_info()

        assert hasattr(unidock_receptor_topology_builder, 'atom_info_nested_list')
        assert (
            len(unidock_receptor_topology_builder.atom_info_nested_list)
            > 0
        )

def test_receptor_topology_dms(receptor_topology_test_dms_file):
    root_temp_dir_name = '/tmp'
    temp_dir_prefix = os.path.join(
        root_temp_dir_name, get_temp_dir_prefix('test_receptor_topology')
    )

    with TemporaryDirectory(prefix=temp_dir_prefix, delete=True) as working_dir_name:
        unidock_receptor_topology_builder = UnidockReceptorTopologyBuilder(
            receptor_topology_test_dms_file,
            prepared_hydrogen=True,
            covalent_residue_atom_info_list=None,
            working_dir_name=working_dir_name,
        )

        unidock_receptor_topology_builder.generate_receptor_topology()
        unidock_receptor_topology_builder.analyze_receptor_topology()
        unidock_receptor_topology_builder.get_summary_receptor_info()

        assert hasattr(unidock_receptor_topology_builder, 'atom_info_nested_list')
        assert (
            len(unidock_receptor_topology_builder.atom_info_nested_list)
            > 0
        )

def test_receptor_topology_RNA_pdb(receptor_topology_RNA_test_pdb_file):
    root_temp_dir_name = '/tmp'
    temp_dir_prefix = os.path.join(
        root_temp_dir_name, get_temp_dir_prefix('test_receptor_topology')
    )

    with TemporaryDirectory(prefix=temp_dir_prefix, delete=True) as working_dir_name:
        unidock_receptor_topology_builder = UnidockReceptorTopologyBuilder(
            receptor_topology_RNA_test_pdb_file,
            prepared_hydrogen=False,
            covalent_residue_atom_info_list=None,
            working_dir_name=working_dir_name,
        )

        unidock_receptor_topology_builder.generate_receptor_topology()
        unidock_receptor_topology_builder.analyze_receptor_topology()
        unidock_receptor_topology_builder.get_summary_receptor_info()

        assert hasattr(unidock_receptor_topology_builder, 'atom_info_nested_list')
        assert (
            len(unidock_receptor_topology_builder.atom_info_nested_list)
            > 0
        )
