import os
import pytest
import yaml

from context import TEST_DATA_DIR

from unidock_processing.io.yaml import read_unidock_params_from_yaml
from unidock_processing.io.get_temp_dir_prefix import get_temp_dir_prefix
from unidock_processing.io.tempfile import TemporaryDirectory
from unidock_processing.unidocktools.unidock_protocol_runner import (
    UnidockProtocolRunner,
)

@pytest.mark.parametrize(
    'receptor,ligand,pocket_center',
    [
        (
            os.path.join(TEST_DATA_DIR, 'free_docking', 'molecular_docking',
                         '1G9V_protein_water_cleaned.pdb'),
            os.path.join(TEST_DATA_DIR, 'free_docking', 'molecular_docking', 'ligand_prepared.sdf'),
            (5.122, 18.327, 37.332)
        ),
        (
            os.path.join(TEST_DATA_DIR, 'free_docking', 'virtual_screening',
                         '5WIU_protein_cleaned.pdb'),
            os.path.join(TEST_DATA_DIR, 'free_docking', 'virtual_screening', 'actives_cleaned.sdf'),
            (-18.0, 15.2, -17.0)
        ),
    ]
)

def test_free_docking(
    receptor,
    ligand,
    pocket_center,
):
    box_size = (30.0, 30.0, 30.0)

    root_temp_dir_name = '/tmp'
    temp_dir_prefix = os.path.join(
        root_temp_dir_name, get_temp_dir_prefix('test_free_docking')
    )

    with TemporaryDirectory(prefix=temp_dir_prefix, delete=True) as working_dir_name:
        docking_pose_sdf_file_name = os.path.join(working_dir_name, 'unidock2_pose.sdf')
        unidock_protocol_runner = UnidockProtocolRunner(
            receptor,
            [ligand],
            target_center=pocket_center,
            box_size=box_size,
            working_dir_name=working_dir_name,
            docking_pose_sdf_file_name=docking_pose_sdf_file_name
        )

        unidock_protocol_runner.run_unidock_protocol()

        assert os.path.exists(unidock_protocol_runner.docking_pose_sdf_file_name)
        assert os.path.getsize(unidock_protocol_runner.docking_pose_sdf_file_name) > 0

@pytest.mark.parametrize(
    'receptor,ligand,pocket_center,configurations_file',
    [
        (
            os.path.join(TEST_DATA_DIR, 'free_docking', 'molecular_docking',
                         '1G9V_protein_water_cleaned.pdb'),
            os.path.join(TEST_DATA_DIR, 'free_docking', 'molecular_docking', 'ligand_prepared.sdf'),
            (5.122, 18.327, 37.332),
            os.path.join(TEST_DATA_DIR, 'free_docking', 'molecular_docking', 'unidock_configurations.yaml')
        )
    ]
)

def test_free_docking_by_yaml(
    receptor,
    ligand,
    pocket_center,
    configurations_file
):
    with open(configurations_file, 'r') as template_configuration_file:
        unidock2_option_dict = yaml.safe_load(template_configuration_file)

    root_temp_dir_name = '/tmp'
    temp_dir_prefix = os.path.join(
        root_temp_dir_name, get_temp_dir_prefix('test_free_docking')
    )

    with TemporaryDirectory(prefix=temp_dir_prefix, delete=True) as working_dir_name:
        docking_pose_sdf_file_name = os.path.join(working_dir_name, 'unidock2_pose.sdf')

        unidock2_option_dict['Required']['receptor'] = receptor
        unidock2_option_dict['Required']['ligand'] = ligand
        unidock2_option_dict['Required']['center'] = list(pocket_center)

        unidock2_option_dict['Preprocessing']['temp_dir_name'] = working_dir_name
        unidock2_option_dict['Preprocessing']['output_docking_pose_sdf_file_name'] = docking_pose_sdf_file_name

        test_configuration_file_name = os.path.join(
            working_dir_name, 'unidock_configurations.yaml'
        )

        with open(test_configuration_file_name, 'w') as test_configuration_file:
            yaml.dump(unidock2_option_dict, test_configuration_file)

        unidock_params = read_unidock_params_from_yaml(test_configuration_file_name)
        unidock_kwargs_dict = unidock_params.to_protocol_kwargs()
        receptor_file_name = unidock_kwargs_dict.pop('receptor', None)
        ligand_sdf_file_name = unidock_kwargs_dict.pop('ligand', None)
        _ = unidock_kwargs_dict.pop('ligand_batch', None)
        target_center = unidock_kwargs_dict.pop('center', None)
        temp_dir_name = unidock_kwargs_dict.pop('temp_dir_name', None)
        docking_pose_sdf_file_name = unidock_kwargs_dict.pop('output_docking_pose_sdf_file_name', None)
        _ = unidock_kwargs_dict.pop('output_receptor_dms_file_name', None)

        unidock_protocol_runner = UnidockProtocolRunner(
            receptor_file_name,
            [ligand_sdf_file_name],
            target_center=tuple(target_center),
            working_dir_name=temp_dir_name,
            docking_pose_sdf_file_name=docking_pose_sdf_file_name,
            **unidock_kwargs_dict,
        )

        unidock_protocol_runner.run_unidock_protocol()

        assert os.path.exists(unidock_protocol_runner.docking_pose_sdf_file_name)
        assert os.path.getsize(unidock_protocol_runner.docking_pose_sdf_file_name) > 0
