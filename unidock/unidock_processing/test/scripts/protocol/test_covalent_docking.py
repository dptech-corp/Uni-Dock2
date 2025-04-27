import os
import shutil
import uuid
import pytest
import yaml

from unidock.unidock_processing.unidocktools.unidock_protocol_runner import UnidockProtocolRunner

@pytest.fixture
def template_configurations():
    return os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'unidock_configurations.yaml')

@pytest.fixture
def receptor():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'covalent_docking', '1EWL', '1EWL_prepared.pdb')

@pytest.fixture
def ligand():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'covalent_docking', '1EWL')
    ligand_sdf_file_name = os.path.join(data_file_dir_name, 'covalent_mol.sdf')
    ligand_sdf_file_name_list = [ligand_sdf_file_name]

    return ligand_sdf_file_name_list

@pytest.fixture
def pocket_center():
    return (8.411, 13.047, 6.811)

@pytest.fixture
def covalent_residue_atom_info_list():
    covalent_residue_atom_info_list = [['A', 'CYS', 25, 'CA'], ['A', 'CYS', 25, 'CB'], ['A', 'CYS', 25, 'SG']]
    return covalent_residue_atom_info_list

def test_covalent_docking(receptor,
                          ligand,
                          covalent_residue_atom_info_list,
                          pocket_center,
                          template_configurations):

    with open(template_configurations, 'r') as template_configuration_file:
        unidock2_option_dict = yaml.safe_load(template_configuration_file)

    box_size = (30.0, 30.0, 30.0)
    working_dir_name = os.path.abspath(f'./tmp-{uuid.uuid4()}')
    os.mkdir(working_dir_name)

    unidock2_option_dict['Settings']['size_x'] = box_size[0]
    unidock2_option_dict['Settings']['size_y'] = box_size[1]
    unidock2_option_dict['Settings']['size_z'] = box_size[2]
    unidock2_option_dict['Preprocessing']['covalent_ligand'] = True
    unidock2_option_dict['Preprocessing']['covalent_residue_atom_info_list'] = covalent_residue_atom_info_list
    unidock2_option_dict['Preprocessing']['preserve_receptor_hydrogen'] = True
    unidock2_option_dict['Preprocessing']['working_dir_name'] = working_dir_name

    test_configuration_file_name = os.path.join(working_dir_name, 'unidock_configurations.yaml')
    with open(test_configuration_file_name, 'w') as test_configuration_file:
        yaml.dump(unidock2_option_dict, test_configuration_file)

    unidock_protocol_runner = UnidockProtocolRunner(receptor,
                                                    ligand,
                                                    target_center=pocket_center,
                                                    option_yaml_file_name=test_configuration_file_name)

    unidock_protocol_runner.run_unidock_protocol()

    assert os.path.exists(unidock_protocol_runner.unidock2_pose_sdf_file_name)
    assert os.path.getsize(unidock_protocol_runner.unidock2_pose_sdf_file_name) > 0

    shutil.rmtree(working_dir_name, ignore_errors=True)
