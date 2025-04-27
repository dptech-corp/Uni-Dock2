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
def receptor_molecular_docking():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'free_docking', 'molecular_docking', '1G9V_protein_water_cleaned.pdb')

@pytest.fixture
def receptor_virtual_screening():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'free_docking', 'virtual_screening', '5WIU_protein_cleaned.pdb')

@pytest.fixture
def ligand_molecular_docking():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'free_docking', 'molecular_docking', 'ligand_prepared.sdf')

@pytest.fixture
def ligand_virtual_screening():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'free_docking', 'virtual_screening', 'actives_cleaned.sdf')

@pytest.fixture
def pocket_center_molecular_docking():
    return (5.122, 18.327, 37.332)

@pytest.fixture
def pocket_center_virtual_screening():
    return (-18.0, 15.2, -17.0)

def test_molecular_docking(receptor_molecular_docking,
                           ligand_molecular_docking,
                           pocket_center_molecular_docking,
                           template_configurations):

    with open(template_configurations, 'r') as template_configuration_file:
        unidock2_option_dict = yaml.safe_load(template_configuration_file)

    box_size = (30.0, 30.0, 30.0)
    working_dir_name = os.path.abspath(f'./tmp-{uuid.uuid4()}')
    os.mkdir(working_dir_name)

    unidock2_option_dict['Settings']['size_x'] = box_size[0]
    unidock2_option_dict['Settings']['size_y'] = box_size[1]
    unidock2_option_dict['Settings']['size_z'] = box_size[2]
    unidock2_option_dict['Preprocessing']['working_dir_name'] = working_dir_name

    test_configuration_file_name = os.path.join(working_dir_name, 'unidock_configurations.yaml')
    with open(test_configuration_file_name, 'w') as test_configuration_file:
        yaml.dump(unidock2_option_dict, test_configuration_file)

    unidock_protocol_runner = UnidockProtocolRunner(receptor_molecular_docking,
                                                    [ligand_molecular_docking],
                                                    target_center=pocket_center_molecular_docking,
                                                    option_yaml_file_name=test_configuration_file_name)

    unidock_protocol_runner.run_unidock_protocol()

    assert os.path.exists(unidock_protocol_runner.unidock2_pose_sdf_file_name)
    assert os.path.getsize(unidock_protocol_runner.unidock2_pose_sdf_file_name) > 0

    shutil.rmtree(working_dir_name, ignore_errors=True)

def test_virtual_screening(receptor_virtual_screening,
                           ligand_virtual_screening,
                           pocket_center_virtual_screening,
                           template_configurations):

    with open(template_configurations, 'r') as template_configuration_file:
        unidock2_option_dict = yaml.safe_load(template_configuration_file)

    box_size = (30.0, 30.0, 30.0)
    working_dir_name = os.path.abspath(f'./tmp-{uuid.uuid4()}')
    os.mkdir(working_dir_name)

    unidock2_option_dict['Settings']['size_x'] = box_size[0]
    unidock2_option_dict['Settings']['size_y'] = box_size[1]
    unidock2_option_dict['Settings']['size_z'] = box_size[2]
    unidock2_option_dict['Preprocessing']['working_dir_name'] = working_dir_name

    test_configuration_file_name = os.path.join(working_dir_name, 'unidock_configurations.yaml')
    with open(test_configuration_file_name, 'w') as test_configuration_file:
        yaml.dump(unidock2_option_dict, test_configuration_file)

    unidock_protocol_runner = UnidockProtocolRunner(receptor_virtual_screening,
                                                    [ligand_virtual_screening],
                                                    target_center=pocket_center_virtual_screening,
                                                    option_yaml_file_name=test_configuration_file_name)

    unidock_protocol_runner.run_unidock_protocol()

    assert os.path.exists(unidock_protocol_runner.unidock2_pose_sdf_file_name)
    assert os.path.getsize(unidock_protocol_runner.unidock2_pose_sdf_file_name) > 0

    shutil.rmtree(working_dir_name, ignore_errors=True)
