import pytest

from unidock2.config import UnidockConfig, UnknownConfigurationWarning


DEFAULT_CONFIG = {
    "Required": {
        "receptor": None,
        "ligand": None,
        "ligand_batch": None,
        "center": [0.0, 0.0, 0.0],
    },
    "Advanced": {
        "exhaustiveness": 512,
        "randomize": True,
        "mc_steps": 40,
        "opt_steps": -1,
        "refine_steps": 5,
        "num_pose": 10,
        "rmsd_limit": 1.0,
        "energy_range": 5.0,
        "seed": 1234567,
        "bias": "no",
        "bias_k": 0.1,
        "use_tor_lib": False,
        "energy_decomp": False,
    },
    "Hardware": {"n_cpu": None, "gpu_device_id": 0, "max_gpu_memory": 0},
    "Settings": {
        "box_size": [30.0, 30.0, 30.0],
        "task": "screen",
        "search_mode": "balance",
    },
    "Preprocessing": {
        "construct_ff": False,
        "template_docking": False,
        "reference_sdf_file_name": None,
        "compute_center": True,
        "core_atom_mapping_dict_list": None,
        "covalent_ligand": False,
        "covalent_residue_atom_info_list": None,
        "preserve_receptor_hydrogen": False,
        "temp_dir_name": "/tmp",
        "engine_checkpoint": False,
        "output_receptor_dms_file_name": "receptor_parameterized.dms",
        "output_docking_pose_sdf_file_name": "unidock2_pose.sdf",
    },
}


def test_default_config_and_flattened_compatibility_snapshots():
    config = UnidockConfig()

    assert config.model_dump(by_alias=True) == DEFAULT_CONFIG

    expected_flat_config = {}
    for section in DEFAULT_CONFIG.values():
        expected_flat_config.update(section)
    assert config.to_protocol_kwargs() == expected_flat_config


def test_yaml_aliases_and_python_section_names_are_supported():
    yaml_shaped = UnidockConfig.from_dict({"Advanced": {"mc_steps": 21}})
    python_shaped = UnidockConfig.from_dict({"advanced": {"mc_steps": 22}})

    assert yaml_shaped.advanced.mc_steps == 21
    assert python_shaped.advanced.mc_steps == 22


def test_unknown_configuration_fields_warn_and_remain_ignored():
    with pytest.warns(UnknownConfigurationWarning) as caught:
        config = UnidockConfig.from_dict(
            {
                "Advanced": {"mc_steps": 23, "mc_stepz": 99},
                "Unsupported": {"option": True},
            }
        )

    messages = {str(warning.message) for warning in caught}
    assert messages == {
        "Unknown configuration field 'Advanced.mc_stepz' will be ignored.",
        "Unknown configuration section 'Unsupported' will be ignored.",
    }
    assert config.advanced.mc_steps == 23
    assert not hasattr(config.advanced, "mc_stepz")


def test_flat_overrides_are_validated_by_the_owning_section():
    config = UnidockConfig().with_overrides(
        center=[1, 2, 3],
        mc_steps="24",
        box_size=(10, 11, 12),
    )

    assert config.required.center == [1.0, 2.0, 3.0]
    assert config.advanced.mc_steps == 24
    assert config.settings.box_size == [10.0, 11.0, 12.0]

    with pytest.raises(TypeError, match="unknown_option"):
        config.with_overrides(unknown_option=True)


@pytest.mark.parametrize("field_name", ["center", "box_size"])
def test_three_coordinate_fields_keep_their_validation(field_name):
    with pytest.raises(ValueError, match="requires 3 elements"):
        UnidockConfig().with_overrides(**{field_name: [1, 2]})
