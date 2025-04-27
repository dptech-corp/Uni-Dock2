import os
import shutil
import uuid
import pytest
import yaml

from rdkit import Chem
from unidock.unidock_processing.unidocktools.unidock_receptor_topology_builder import UnidockReceptorTopologyBuilder

@pytest.fixture
def template_configurations():
    return os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'unidock_configurations.yaml')

@pytest.fixture
def receptor_topology_test_pdb_file():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'receptor_topology')
    receptor_pdb_file_name = os.path.join(data_file_dir_name, 'test_receptor_topology_protocol.pdb')

    return receptor_pdb_file_name

@pytest.fixture
def receptor_topology_test_dms_file():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'receptor_topology')
    receptor_dms_file_name = os.path.join(data_file_dir_name, 'test_receptor_topology_protocol.dms')

    return receptor_dms_file_name

@pytest.fixture
def receptor_topology_RNA_test_pdb_file():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'receptor_topology')
    receptor_pdb_file_name = os.path.join(data_file_dir_name, 'test_receptor_topology_RNA.pdb')

    return receptor_pdb_file_name

def test_receptor_topology_pdb(receptor_topology_test_pdb_file):
    working_dir_name = os.path.abspath(f'./tmp-{uuid.uuid4()}')
    os.mkdir(working_dir_name)

    unidock_receptor_topology_builder = UnidockReceptorTopologyBuilder(receptor_topology_test_pdb_file,
                                                                       prepared_hydrogen=True,
                                                                       covalent_residue_atom_info_list=None,
                                                                       working_dir_name=working_dir_name)

    unidock_receptor_topology_builder.generate_receptor_topology()
    unidock_receptor_topology_builder.get_summary_receptor_info_dict()

    assert hasattr(unidock_receptor_topology_builder, 'receptor_info_summary_dict')
    assert len(unidock_receptor_topology_builder.receptor_info_summary_dict['receptor']) > 0

def test_receptor_topology_dms(receptor_topology_test_dms_file):
    working_dir_name = os.path.abspath(f'./tmp-{uuid.uuid4()}')
    os.mkdir(working_dir_name)

    unidock_receptor_topology_builder = UnidockReceptorTopologyBuilder(receptor_topology_test_dms_file,
                                                                       prepared_hydrogen=True,
                                                                       covalent_residue_atom_info_list=None,
                                                                       working_dir_name=working_dir_name)

    unidock_receptor_topology_builder.generate_receptor_topology()
    unidock_receptor_topology_builder.get_summary_receptor_info_dict()

    assert hasattr(unidock_receptor_topology_builder, 'receptor_info_summary_dict')
    assert len(unidock_receptor_topology_builder.receptor_info_summary_dict['receptor']) > 0

def test_receptor_topology_RNA_pdb(receptor_topology_RNA_test_pdb_file):
    working_dir_name = os.path.abspath(f'./tmp-{uuid.uuid4()}')
    os.mkdir(working_dir_name)

    unidock_receptor_topology_builder = UnidockReceptorTopologyBuilder(receptor_topology_RNA_test_pdb_file,
                                                                       prepared_hydrogen=False,
                                                                       covalent_residue_atom_info_list=None,
                                                                       working_dir_name=working_dir_name)

    unidock_receptor_topology_builder.generate_receptor_topology()
    unidock_receptor_topology_builder.get_summary_receptor_info_dict()

    assert hasattr(unidock_receptor_topology_builder, 'receptor_info_summary_dict')
    assert len(unidock_receptor_topology_builder.receptor_info_summary_dict['receptor']) > 0