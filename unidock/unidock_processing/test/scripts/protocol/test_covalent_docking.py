import os
import pytest

from context import TEST_DATA_DIR

from unidock_processing.io.get_temp_dir_prefix import get_temp_dir_prefix
from unidock_processing.io.tempfile import TemporaryDirectory
from unidock_processing.unidocktools.unidock_protocol_runner import (
    UnidockProtocolRunner,
)

@pytest.mark.parametrize(
    'receptor,ligand,covalent_residue_atom_info_list,pocket_center',
    [
        (
            os.path.join(TEST_DATA_DIR, 'covalent_docking', '1EWL', '1EWL_prepared.pdb'),
            [os.path.join(TEST_DATA_DIR, 'covalent_docking', '1EWL', 'covalent_mol.sdf')],
            [
                ['', 'CYX', 25, 'CA'],
                ['', 'CYX', 25, 'CB'],
                ['', 'CYX', 25, 'SG'],
            ],
            (8.411, 13.047, 6.811),
        ),
    ]
)

def test_covalent_docking(
    receptor,
    ligand,
    covalent_residue_atom_info_list,
    pocket_center,
):
    box_size = (30.0, 30.0, 30.0)
    root_temp_dir_name = '/tmp'
    temp_dir_prefix = os.path.join(
        root_temp_dir_name, get_temp_dir_prefix('test_covalent_docking')
    )

    with TemporaryDirectory(prefix=temp_dir_prefix, delete=True) as working_dir_name:
        docking_pose_sdf_file_name = os.path.join(working_dir_name, 'unidock2_pose.sdf')
        unidock_protocol_runner = UnidockProtocolRunner(
            receptor,
            target_center=pocket_center,
            box_size=box_size,
            ligand_sdf_file_name_list=ligand,
            covalent_ligand=True,
            covalent_residue_atom_info_list=covalent_residue_atom_info_list,
            preserve_receptor_hydrogen=True,
            working_dir_name=working_dir_name,
            docking_pose_sdf_file_name=docking_pose_sdf_file_name
        )

        unidock_protocol_runner.run_unidock_protocol()

        assert os.path.exists(unidock_protocol_runner.docking_pose_sdf_file_name)
        assert os.path.getsize(unidock_protocol_runner.docking_pose_sdf_file_name) > 0
