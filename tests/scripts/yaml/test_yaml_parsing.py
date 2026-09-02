import os

import pytest

from context import TEST_DATA_DIR

from unidock2.config import UnidockConfig
from unidock2.io.yaml import read_unidock_params_from_yaml


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
    """Values in the file win; fields the file omits fall back to the schema.

    The default values themselves are covered by
    `tests/scripts/config/test_config_models.py`.
    """

    yaml_params = read_unidock_params_from_yaml(configurations_file)
    defaults = UnidockConfig()

    # Written in the YAML file, and different from the schema default.
    assert yaml_params.required.receptor == '1G9V_protein_water_cleaned.pdb'
    assert yaml_params.required.ligand == 'ligand_prepared.sdf'
    assert yaml_params.required.center == [5.122, 18.327, 37.332]
    assert yaml_params.advanced.mc_steps == 20
    assert yaml_params.advanced.energy_range == 3.0
    assert yaml_params.advanced.seed == 12345

    # Absent from the YAML file, so the schema default applies. engine_checkpoint
    # guards against an omitted field silently becoming falsy.
    assert yaml_params.advanced.bias == defaults.advanced.bias
    assert yaml_params.advanced.energy_decomp == defaults.advanced.energy_decomp
    assert yaml_params.hardware.n_cpu == defaults.hardware.n_cpu
    assert yaml_params.hardware.max_gpu_memory == defaults.hardware.max_gpu_memory
    assert yaml_params.preprocessing.construct_ff == defaults.preprocessing.construct_ff
    assert yaml_params.preprocessing.keep_workdir == defaults.preprocessing.keep_workdir
    assert yaml_params.preprocessing.engine_checkpoint is True

    # The flattened view exposes exactly the schema fields, whatever the file holds.
    assert set(yaml_params.to_protocol_kwargs()) == set(defaults.to_protocol_kwargs())
