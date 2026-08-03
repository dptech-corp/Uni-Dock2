import os
import pytest

from rdkit import Chem
from rdkit.Chem import rdMolAlign

from context import TEST_DATA_DIR

from unidock2.io.get_temp_dir_prefix import get_temp_dir_prefix
from unidock2.io.tempfile import TemporaryDirectory
from unidock2.unidocktools.unidock_protocol_runner import (
    UnidockProtocolRunner,
)

TOP1_RMSD_LIMIT = 2.0


def calc_rmsd(ref_ligand, target_ligand):
    ref_mol = Chem.SDMolSupplier(str(ref_ligand), removeHs=True)[0]
    target_mols = Chem.SDMolSupplier(str(target_ligand), removeHs=True)
    return [rdMolAlign.CalcRMS(ref_mol, target_mol) for target_mol in target_mols]


@pytest.mark.parametrize(
    'receptor,ligand,reference,covalent_residue_atom_info_list,pocket_center',
    [
        (
            os.path.join(TEST_DATA_DIR, 'covalent_docking', '1EWL', '1EWL_prepared.pdb'),
            [os.path.join(TEST_DATA_DIR, 'covalent_docking', '1EWL', 'covalent_mol.sdf')],
            os.path.join(TEST_DATA_DIR, 'covalent_docking', '1EWL', '1EWL_ligand.sdf'),
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
    reference,
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
            ligand,
            target_center=pocket_center,
            box_size=box_size,
            covalent_ligand=True,
            covalent_residue_atom_info_list=covalent_residue_atom_info_list,
            preserve_receptor_hydrogen=True,
            working_dir_name=working_dir_name,
            docking_pose_sdf_file_name=docking_pose_sdf_file_name
        )

        unidock_protocol_runner.run_unidock_protocol()

        assert os.path.exists(unidock_protocol_runner.docking_pose_sdf_file_name)
        assert os.path.getsize(unidock_protocol_runner.docking_pose_sdf_file_name) > 0

        rmsd_list = calc_rmsd(
            reference,
            unidock_protocol_runner.docking_pose_sdf_file_name,
        )
        assert rmsd_list, 'No docking poses found in output SDF.'
        assert rmsd_list[0] < TOP1_RMSD_LIMIT, (
            f'Top-1 RMSD {rmsd_list[0]:.3f} A exceeds '
            f'{TOP1_RMSD_LIMIT:.1f} A (best RMSD: {min(rmsd_list):.3f} A).'
        )
