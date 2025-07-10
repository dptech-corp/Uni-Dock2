import os
import pytest

from context import TEST_DATA_DIR

from unidock_processing.io.yaml import read_unidock_params_from_yaml

@pytest.mark.parametrize(
    'configurations_file',
    [
        (
            os.path.join(TEST_DATA_DIR, 'yaml_configurations', 'unidock_configurations.yaml')
        )
    ]
)

def test_yaml_parsing(
    configurations_file,
):

    yaml_params = read_unidock_params_from_yaml(configurations_file)

    assert yaml_params.required.receptor == '1G9V_protein_water_cleaned.pdb'
    assert yaml_params.required.ligand == 'ligand_prepared.sdf'
    assert yaml_params.required.ligand_batch is None
    assert yaml_params.required.center == (5.122, 18.327, 37.332)

    assert yaml_params.advanced.exhaustiveness == 512
    assert yaml_params.advanced.randomize
    assert yaml_params.advanced.mc_steps == 20
    assert yaml_params.advanced.opt_steps == -1
    assert yaml_params.advanced.refine_steps == 5
    assert yaml_params.advanced.num_pose == 10
    assert yaml_params.advanced.rmsd_limit == 1.0
    assert yaml_params.advanced.energy_range == 3.0
    assert yaml_params.advanced.seed == 12345
    assert not yaml_params.advanced.use_tor_lib

    assert yaml_params.hardware.n_cpu is None
    assert yaml_params.hardware.gpu_device_id == 0

    assert yaml_params.settings.box_size == (30.0, 30.0, 30.0)
    assert yaml_params.settings.task == 'screen'
    assert yaml_params.settings.search_mode == 'balance'

    assert not yaml_params.preprocessing.template_docking
    assert yaml_params.preprocessing.reference_sdf_file_name is None
    assert yaml_params.preprocessing.core_atom_mapping_dict_list is None
    assert not yaml_params.preprocessing.covalent_ligand
    assert yaml_params.preprocessing.covalent_residue_atom_info_list is None
    assert not yaml_params.preprocessing.preserve_receptor_hydrogen
    assert yaml_params.preprocessing.temp_dir_name == '/tmp'
    assert yaml_params.preprocessing.output_receptor_dms_file_name == 'receptor_parameterized.dms'
    assert yaml_params.preprocessing.output_docking_pose_sdf_file_name == 'unidock2_pose.sdf'

    valid_configurations_dict = {'receptor': '1G9V_protein_water_cleaned.pdb',
                                 'ligand': 'ligand_prepared.sdf',
                                 'ligand_batch': None,
                                 'center': (5.122, 18.327, 37.332),
                                 'exhaustiveness': 512,
                                 'randomize': True,
                                 'mc_steps': 20,
                                 'opt_steps': -1,
                                 'refine_steps': 5,
                                 'num_pose': 10,
                                 'rmsd_limit': 1.0,
                                 'energy_range': 3.0,
                                 'seed': 12345,
                                 'use_tor_lib': False,
                                 'n_cpu': None,
                                 'gpu_device_id': 0,
                                 'box_size': (30.0, 30.0, 30.0),
                                 'task': 'screen',
                                 'search_mode': 'balance',
                                 'template_docking': False,
                                 'reference_sdf_file_name': None,
                                 'core_atom_mapping_dict_list': None,
                                 'covalent_ligand': False,
                                 'covalent_residue_atom_info_list': None,
                                 'preserve_receptor_hydrogen': False,
                                 'temp_dir_name': '/tmp',
                                 'output_receptor_dms_file_name': 'receptor_parameterized.dms',
                                 'output_docking_pose_sdf_file_name': 'unidock2_pose.sdf'}

    configurations_dict = yaml_params.to_protocol_kwargs()
    assert configurations_dict == valid_configurations_dict
