import os
import shutil
from glob import glob
import uuid
import pytest

from unidock_processing.unidocktools.unidock_protocol_runner import (
    UnidockProtocolRunner,
)


TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "constraint_docking"
)


@pytest.mark.parametrize(
    "receptor,ligand,reference,core_atom_mapping,pocket_center",
    [
        (
            os.path.join(TEST_DATA_DIR, "automatic_atom_mapping", "Bace.pdb"),
            glob(os.path.join(TEST_DATA_DIR, "automatic_atom_mapping", "CAT*.sdf")),
            os.path.join(TEST_DATA_DIR, "automatic_atom_mapping", "reference.sdf"),
            None,
            (14.786, -0.626, -1.088)
        ),
        (
            os.path.join(TEST_DATA_DIR, "manual_atom_mapping", "protein.pdb"),
            [os.path.join(TEST_DATA_DIR, "manual_atom_mapping", "ligand.sdf")],
            os.path.join(TEST_DATA_DIR, "manual_atom_mapping", "reference.sdf"),
            [{
                "0": 14,
                "1": 15,
                "10": 11,
                "11": 12,
                "12": 13,
                "16": 1,
                "17": 2,
                "18": 3,
                "19": 4,
                "20": 6,
                "21": 7,
                "22": 8,
                "23": 9,
                "24": 20,
                "25": 21,
                "27": 27,
                "6": 24,
                "7": 0,
                "8": 5,
                "9": 10,
            }],
            (9.028, 0.804, 21.789)
        ),
        (
            os.path.join(TEST_DATA_DIR, "RNA_case", "test_receptor_topology_RNA.pdb"),
            [os.path.join(TEST_DATA_DIR, "RNA_case", "ligand.sdf")],
            os.path.join(TEST_DATA_DIR, "RNA_case", "reference.sdf"),
            None,
            (-12.097, 34.025, 1278.974)
        ),
    ],
)
def test_constraint_docking(
    receptor,
    ligand,
    reference,
    core_atom_mapping,
    pocket_center,
):
    box_size = (30.0, 30.0, 30.0)
    working_dir_name = os.path.abspath(f"./tmp-{uuid.uuid4()}")
    docking_pose_sdf_file_name = os.path.join(working_dir_name, 'unidock2_pose.sdf')
    os.mkdir(working_dir_name)

    unidock_protocol_runner = UnidockProtocolRunner(
        receptor,
        ligand,
        target_center=pocket_center,
        box_size=box_size,
        template_docking=True,
        reference_sdf_file_name=reference,
        preserve_receptor_hydrogen=True,
        working_dir_name=working_dir_name,
        docking_pose_sdf_file_name=docking_pose_sdf_file_name,
        core_atom_mapping_dict_list=core_atom_mapping
    )

    unidock_protocol_runner.run_unidock_protocol()

    assert os.path.exists(unidock_protocol_runner.docking_pose_sdf_file_name)
    assert os.path.getsize(unidock_protocol_runner.docking_pose_sdf_file_name) > 0

    shutil.rmtree(working_dir_name, ignore_errors=True)
