import os
import shutil
import uuid
import pytest
import yaml

from unidock_processing.io import read_unidock_params_from_yaml
from unidock_processing.unidocktools.unidock_protocol_runner import (
    UnidockProtocolRunner,
)


TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "free_docking"
)


@pytest.mark.parametrize(
    "receptor,ligand,pocket_center",
    [
        (
            os.path.join(TEST_DATA_DIR, "molecular_docking",
                         "1G9V_protein_water_cleaned.pdb"),
            os.path.join(TEST_DATA_DIR, "molecular_docking", "ligand_prepared.sdf"),
            (5.122, 18.327, 37.332)
        ),
        (
            os.path.join(TEST_DATA_DIR, "virtual_screening",
                         "5WIU_protein_cleaned.pdb"),
            os.path.join(TEST_DATA_DIR, "virtual_screening", "actives_cleaned.sdf"),
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
    working_dir_name = os.path.abspath(f"./tmp-{uuid.uuid4()}")
    docking_pose_sdf_file_name = os.path.join(working_dir_name, 'unidock2_pose.sdf')
    os.mkdir(working_dir_name)

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

    shutil.rmtree(working_dir_name, ignore_errors=True)


@pytest.mark.parametrize(
    "receptor,ligand,pocket_center,configurations_file",
    [
        (
            os.path.join(TEST_DATA_DIR, "molecular_docking",
                         "1G9V_protein_water_cleaned.pdb"),
            os.path.join(TEST_DATA_DIR, "molecular_docking", "ligand_prepared.sdf"),
            (5.122, 18.327, 37.332),
            os.path.join(TEST_DATA_DIR, "unidock_configurations.yaml")
        )
    ]
)

def test_free_docking_by_yaml(
    receptor,
    ligand,
    pocket_center,
    configurations_file,
):
    with open(configurations_file, "r") as template_configuration_file:
        unidock2_option_dict = yaml.safe_load(template_configuration_file)

    working_dir_name = os.path.abspath(f"./tmp-{uuid.uuid4()}")
    docking_pose_sdf_file_name = os.path.join(working_dir_name, 'unidock2_pose.sdf')
    os.mkdir(working_dir_name)

    unidock2_option_dict["Preprocessing"]["temp_dir_name"] = working_dir_name
    unidock2_option_dict["Preprocessing"]["output_docking_pose_sdf_file_name"] = docking_pose_sdf_file_name

    test_configuration_file_name = os.path.join(
        working_dir_name, "unidock_configurations.yaml"
    )
    with open(test_configuration_file_name, "w") as test_configuration_file:
        yaml.dump(unidock2_option_dict, test_configuration_file)

    unidock_params = read_unidock_params_from_yaml(test_configuration_file_name)
    unidock_protocol_runner = UnidockProtocolRunner(
        receptor,
        [ligand],
        target_center=pocket_center,
        **unidock_params.to_protocol_kwargs()
    )

    unidock_protocol_runner.run_unidock_protocol()

    assert os.path.exists(unidock_protocol_runner.docking_pose_sdf_file_name)
    assert os.path.getsize(unidock_protocol_runner.docking_pose_sdf_file_name) > 0

    shutil.rmtree(working_dir_name, ignore_errors=True)
