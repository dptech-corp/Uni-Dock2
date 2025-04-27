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
def receptor_automatic_case():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking', 'automatic_atom_mapping', 'Bace.pdb')

@pytest.fixture
def ligand_automatic_case():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking', 'automatic_atom_mapping')
    data_file_name_list = os.listdir(data_file_dir_name)

    ligand_sdf_file_name_list = []
    for data_file_name in data_file_name_list:
        if data_file_name.startswith('CAT') and data_file_name.endswith('sdf'):
            ligand_sdf_file_name = os.path.join(data_file_dir_name, data_file_name)
            ligand_sdf_file_name_list.append(ligand_sdf_file_name)

    return ligand_sdf_file_name_list

@pytest.fixture
def reference_automatic_case():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking', 'automatic_atom_mapping', 'reference.sdf')

@pytest.fixture
def pocket_center_automatic_case():
    return (14.786, -0.626, -1.088)

@pytest.fixture
def receptor_manual_case():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking', 'manual_atom_mapping', 'protein.pdb')

@pytest.fixture
def ligand_manual_case():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking', 'manual_atom_mapping')
    ligand_sdf_file_name = os.path.join(data_file_dir_name, 'ligand.sdf')
    ligand_sdf_file_name_list = [ligand_sdf_file_name]

    return ligand_sdf_file_name_list

@pytest.fixture
def reference_manual_case():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking', 'manual_atom_mapping', 'reference.sdf')

@pytest.fixture
def pocket_center_manual_case():
    return (9.028, 0.804, 21.789)

@pytest.fixture
def core_atom_mapping_manual_case():
    core_atom_mapping_dict = {"0":14, "1":15, "10":11, "11":12, "12":13, "16":1, "17":2, "18":3, "19":4, "20":6, "21":7, "22":8, "23":9, "24":20, "25":21, "27":27, "6":24, "7":0, "8":5, "9":10}
    return [core_atom_mapping_dict]

@pytest.fixture
def receptor_RNA_case():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking', 'RNA_case', 'test_receptor_topology_RNA.pdb')

@pytest.fixture
def ligand_RNA_case():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking', 'RNA_case')
    ligand_sdf_file_name = os.path.join(data_file_dir_name, 'ligand.sdf')
    ligand_sdf_file_name_list = [ligand_sdf_file_name]

    return ligand_sdf_file_name_list

@pytest.fixture
def reference_RNA_case():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'constraint_docking', 'RNA_case', 'reference.sdf')

@pytest.fixture
def pocket_center_RNA_case():
    return (-12.097, 34.025, 1278.974)

def test_automatic_case(receptor_automatic_case,
                        ligand_automatic_case,
                        reference_automatic_case,
                        pocket_center_automatic_case,
                        template_configurations):

    with open(template_configurations, 'r') as template_configuration_file:
        unidock2_option_dict = yaml.safe_load(template_configuration_file)

    box_size = (30.0, 30.0, 30.0)
    working_dir_name = os.path.abspath(f'./tmp-{uuid.uuid4()}')
    os.mkdir(working_dir_name)

    unidock2_option_dict['Settings']['size_x'] = box_size[0]
    unidock2_option_dict['Settings']['size_y'] = box_size[1]
    unidock2_option_dict['Settings']['size_z'] = box_size[2]
    unidock2_option_dict['Preprocessing']['template_docking'] = True
    unidock2_option_dict['Preprocessing']['reference_sdf_file_name'] = reference_automatic_case
    unidock2_option_dict['Preprocessing']['core_atom_mapping_dict_list'] = None
    unidock2_option_dict['Preprocessing']['working_dir_name'] = working_dir_name

    test_configuration_file_name = os.path.join(working_dir_name, 'unidock_configurations.yaml')
    with open(test_configuration_file_name, 'w') as test_configuration_file:
        yaml.dump(unidock2_option_dict, test_configuration_file)

    unidock_protocol_runner = UnidockProtocolRunner(receptor_automatic_case,
                                                    ligand_automatic_case,
                                                    target_center=pocket_center_automatic_case,
                                                    option_yaml_file_name=test_configuration_file_name)

    unidock_protocol_runner.run_unidock_protocol()

    assert os.path.exists(unidock_protocol_runner.unidock2_pose_sdf_file_name)
    assert os.path.getsize(unidock_protocol_runner.unidock2_pose_sdf_file_name) > 0

    shutil.rmtree(working_dir_name, ignore_errors=True)

def test_manual_case(receptor_manual_case,
                     ligand_manual_case,
                     reference_manual_case,
                     core_atom_mapping_manual_case,
                     pocket_center_manual_case,
                     template_configurations):

    with open(template_configurations, 'r') as template_configuration_file:
        unidock2_option_dict = yaml.safe_load(template_configuration_file)

    box_size = (30.0, 30.0, 30.0)
    working_dir_name = os.path.abspath(f'./tmp-{uuid.uuid4()}')
    os.mkdir(working_dir_name)

    unidock2_option_dict['Settings']['size_x'] = box_size[0]
    unidock2_option_dict['Settings']['size_y'] = box_size[1]
    unidock2_option_dict['Settings']['size_z'] = box_size[2]
    unidock2_option_dict['Preprocessing']['template_docking'] = True
    unidock2_option_dict['Preprocessing']['reference_sdf_file_name'] = reference_manual_case
    unidock2_option_dict['Preprocessing']['core_atom_mapping_dict_list'] = core_atom_mapping_manual_case
    unidock2_option_dict['Preprocessing']['preserve_receptor_hydrogen'] = True
    unidock2_option_dict['Preprocessing']['working_dir_name'] = working_dir_name

    test_configuration_file_name = os.path.join(working_dir_name, 'unidock_configurations.yaml')
    with open(test_configuration_file_name, 'w') as test_configuration_file:
        yaml.dump(unidock2_option_dict, test_configuration_file)

    unidock_protocol_runner = UnidockProtocolRunner(receptor_manual_case,
                                                    ligand_manual_case,
                                                    target_center=pocket_center_manual_case,
                                                    option_yaml_file_name=test_configuration_file_name)

    unidock_protocol_runner.run_unidock_protocol()

    assert os.path.exists(unidock_protocol_runner.unidock2_pose_sdf_file_name)
    assert os.path.getsize(unidock_protocol_runner.unidock2_pose_sdf_file_name) > 0

    shutil.rmtree(working_dir_name, ignore_errors=True)

def test_RNA_case(receptor_RNA_case,
                  ligand_RNA_case,
                  reference_RNA_case,
                  pocket_center_RNA_case,
                  template_configurations):

    with open(template_configurations, 'r') as template_configuration_file:
        unidock2_option_dict = yaml.safe_load(template_configuration_file)

    box_size = (30.0, 30.0, 30.0)
    working_dir_name = os.path.abspath(f'./tmp-{uuid.uuid4()}')
    os.mkdir(working_dir_name)

    unidock2_option_dict['Settings']['size_x'] = box_size[0]
    unidock2_option_dict['Settings']['size_y'] = box_size[1]
    unidock2_option_dict['Settings']['size_z'] = box_size[2]
    unidock2_option_dict['Preprocessing']['template_docking'] = True
    unidock2_option_dict['Preprocessing']['reference_sdf_file_name'] = reference_RNA_case
    unidock2_option_dict['Preprocessing']['core_atom_mapping_dict_list'] = None
    unidock2_option_dict['Preprocessing']['working_dir_name'] = working_dir_name

    test_configuration_file_name = os.path.join(working_dir_name, 'unidock_configurations.yaml')
    with open(test_configuration_file_name, 'w') as test_configuration_file:
        yaml.dump(unidock2_option_dict, test_configuration_file)

    unidock_protocol_runner = UnidockProtocolRunner(receptor_RNA_case,
                                                    ligand_RNA_case,
                                                    target_center=pocket_center_RNA_case,
                                                    option_yaml_file_name=test_configuration_file_name)

    unidock_protocol_runner.run_unidock_protocol()

    assert os.path.exists(unidock_protocol_runner.unidock2_pose_sdf_file_name)
    assert os.path.getsize(unidock_protocol_runner.unidock2_pose_sdf_file_name) > 0

    shutil.rmtree(working_dir_name, ignore_errors=True)
