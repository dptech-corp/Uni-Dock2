import math
import os

import pytest

from context import TEST_DATA_DIR

from unidock2.unidocktools.unidock_ligand_topology_builder import (
    UnidockLigandTopologyBuilder,
)
from unidock2.unidocktools.unidock_receptor_topology_builder import (
    UnidockReceptorTopologyBuilder,
)


RESERVED_TOP_LEVEL_KEYS = {"receptor", "score"}
REQUIRED_LIGAND_KEYS = {"atoms", "torsions", "root_atoms"}
OPTIONAL_LIGAND_KEYS = {"fragment_atom_idx"}


def _require(condition, path, message):
    assert condition, f"{path}: {message}"


def _require_integer(value, path):
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        path,
        "expected an integer",
    )


def _require_finite_number(value, path):
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value),
        path,
        "expected a finite number",
    )


def _validate_atom_indices(indices, path, num_atoms, current_atom_idx=None):
    _require(isinstance(indices, list), path, "expected a list")
    for index_idx, atom_idx in enumerate(indices):
        atom_path = f"{path}[{index_idx}]"
        _require_integer(atom_idx, atom_path)
        _require(0 <= atom_idx < num_atoms, atom_path, "atom index out of range")
        if current_atom_idx is not None:
            _require(atom_idx != current_atom_idx, atom_path, "self pair is not allowed")


def _validate_receptor(receptor):
    _require(isinstance(receptor, list), "receptor", "expected a list")
    _require(receptor, "receptor", "must contain at least one atom")

    for atom_idx, atom in enumerate(receptor):
        path = f"receptor[{atom_idx}]"
        _require(isinstance(atom, (list, tuple)), path, "expected an atom sequence")
        _require(len(atom) == 6, path, "expected 6 items")

        for value_idx in (0, 1, 2, 5):
            _require_finite_number(atom[value_idx], f"{path}[{value_idx}]")

        _require_integer(atom[3], f"{path}[3]")
        _require(0 <= atom[3] <= 20, f"{path}[3]", "vina type must be within [0, 20]")
        _require_integer(atom[4], f"{path}[4]")
        _require(atom[4] >= 0, f"{path}[4]", "FF atom type must be non-negative")


def _validate_ligand_atom(atom, atom_idx, num_atoms, ligand_path):
    path = f"{ligand_path}.atoms[{atom_idx}]"
    _require(isinstance(atom, (list, tuple)), path, "expected an atom sequence")
    _require(len(atom) in (8, 9), path, "expected 8 or 9 items")

    for value_idx in (0, 1, 2, 5):
        _require_finite_number(atom[value_idx], f"{path}[{value_idx}]")

    _require_integer(atom[3], f"{path}[3]")
    _require(0 <= atom[3] <= 20, f"{path}[3]", "vina type must be within [0, 20]")
    _require_integer(atom[4], f"{path}[4]")
    _require(atom[4] >= 0, f"{path}[4]", "FF atom type must be non-negative")

    _validate_atom_indices(atom[6], f"{path}[6]", num_atoms, atom_idx)
    _validate_atom_indices(atom[7], f"{path}[7]", num_atoms, atom_idx)
    _require(
        set(atom[6]).isdisjoint(atom[7]),
        path,
        "1-4 pairs must not overlap 1-2/1-3 pairs",
    )

    if len(atom) == 9:
        biases = atom[8]
        _require(isinstance(biases, list), f"{path}[8]", "expected a list")
        for bias_idx, bias in enumerate(biases):
            bias_path = f"{path}[8][{bias_idx}]"
            _require(isinstance(bias, (list, tuple)), bias_path, "expected a bias sequence")
            _require(len(bias) == 5, bias_path, "expected 5 items")
            for value_idx, value in enumerate(bias):
                _require_finite_number(value, f"{bias_path}[{value_idx}]")
            _require(bias[4] > 0, f"{bias_path}[4]", "r2 must be greater than zero")


def _validate_torsion(torsion, torsion_idx, num_atoms, ligand_path):
    path = f"{ligand_path}.torsions[{torsion_idx}]"
    _require(isinstance(torsion, (list, tuple)), path, "expected a torsion sequence")
    _require(len(torsion) == 5, path, "expected 5 items")

    torsion_atoms = torsion[0]
    _require(
        isinstance(torsion_atoms, (list, tuple)),
        f"{path}[0]",
        "expected an atom sequence",
    )
    _require(len(torsion_atoms) == 4, f"{path}[0]", "expected 4 atom indices")
    _validate_atom_indices(list(torsion_atoms), f"{path}[0]", num_atoms)

    _require_finite_number(torsion[1], f"{path}[1]")
    _require(
        -180 <= torsion[1] <= 180,
        f"{path}[1]",
        "dihedral angle must be within [-180, 180]",
    )

    ranges = torsion[2]
    _require(isinstance(ranges, list), f"{path}[2]", "expected a list")
    for range_idx, angle_range in enumerate(ranges):
        range_path = f"{path}[2][{range_idx}]"
        _require(
            isinstance(angle_range, (list, tuple)),
            range_path,
            "expected an angle range",
        )
        _require(len(angle_range) == 2, range_path, "expected 2 items")
        for value_idx, value in enumerate(angle_range):
            _require_finite_number(value, f"{range_path}[{value_idx}]")
            _require(
                -180 <= value <= 180,
                f"{range_path}[{value_idx}]",
                "range endpoint must be within [-180, 180]",
            )
        lower, upper = angle_range
        _require(
            lower < upper or (lower > 0 and upper < 0),
            range_path,
            "invalid torsion range",
        )

    _validate_atom_indices(torsion[3], f"{path}[3]", num_atoms)

    gaff2_parameters = torsion[4]
    _require(isinstance(gaff2_parameters, list), f"{path}[4]", "expected a list")
    for parameter_idx, parameter in enumerate(gaff2_parameters):
        parameter_path = f"{path}[4][{parameter_idx}]"
        _require(
            isinstance(parameter, (list, tuple)),
            parameter_path,
            "expected a parameter sequence",
        )
        _require(len(parameter) == 4, parameter_path, "expected 4 items")
        for value_idx, value in enumerate(parameter):
            _require_finite_number(value, f"{parameter_path}[{value_idx}]")


def _validate_ligand(ligand_name, ligand):
    path = ligand_name
    _require(isinstance(ligand_name, str) and ligand_name, path, "invalid ligand name")
    _require(isinstance(ligand, dict), path, "expected an object")

    ligand_keys = set(ligand)
    missing_keys = REQUIRED_LIGAND_KEYS - ligand_keys
    unknown_keys = ligand_keys - REQUIRED_LIGAND_KEYS - OPTIONAL_LIGAND_KEYS
    _require(not missing_keys, path, f"missing required fields: {sorted(missing_keys)}")
    _require(not unknown_keys, path, f"unknown fields: {sorted(unknown_keys)}")

    atoms = ligand["atoms"]
    _require(isinstance(atoms, list), f"{path}.atoms", "expected a list")
    _require(atoms, f"{path}.atoms", "must contain at least one atom")
    num_atoms = len(atoms)

    for atom_idx, atom in enumerate(atoms):
        _validate_ligand_atom(atom, atom_idx, num_atoms, path)

    root_atoms = ligand["root_atoms"]
    _require(isinstance(root_atoms, list), f"{path}.root_atoms", "expected a list")
    _require(root_atoms, f"{path}.root_atoms", "must not be empty")
    _validate_atom_indices(root_atoms, f"{path}.root_atoms", num_atoms)
    _require(
        len(root_atoms) == len(set(root_atoms)),
        f"{path}.root_atoms",
        "duplicate atom indices are not allowed",
    )

    torsions = ligand["torsions"]
    _require(isinstance(torsions, list), f"{path}.torsions", "expected a list")
    for torsion_idx, torsion in enumerate(torsions):
        _validate_torsion(torsion, torsion_idx, num_atoms, path)

    if "fragment_atom_idx" in ligand:
        fragments = ligand["fragment_atom_idx"]
        _require(
            isinstance(fragments, list),
            f"{path}.fragment_atom_idx",
            "expected a list",
        )
        for fragment_idx, fragment in enumerate(fragments):
            _validate_atom_indices(
                fragment,
                f"{path}.fragment_atom_idx[{fragment_idx}]",
                num_atoms,
            )


def assert_valid_engine_input(data):
    _require(isinstance(data, dict), "root", "expected an object")
    _require("receptor" in data, "receptor", "missing required field")
    _validate_receptor(data["receptor"])

    if "score" in data:
        score = data["score"]
        _require(isinstance(score, list), "score", "expected a list")
        for score_idx, score_name in enumerate(score):
            _require(isinstance(score_name, str), f"score[{score_idx}]", "expected a string")

    ligand_names = [key for key in data if key not in RESERVED_TOP_LEVEL_KEYS]
    _require(ligand_names, "root", "must contain at least one ligand")
    for ligand_name in ligand_names:
        _validate_ligand(ligand_name, data[ligand_name])


def minimal_engine_input():
    return {
        "receptor": [[0.0, 0.0, 0.0, 2, 0, 0.0]],
        "MOL_0": {
            "atoms": [
                [3.0, 0.0, 0.0, 2, 0, 0.0, [1], []],
                [4.5, 0.0, 0.0, 2, 0, 0.0, [0], []],
            ],
            "torsions": [],
            "root_atoms": [0, 1],
        },
    }


def test_minimal_engine_input_is_valid():
    assert_valid_engine_input(minimal_engine_input())


def test_processing_builders_generate_valid_engine_input(tmp_path):
    receptor_file_name = os.path.join(
        TEST_DATA_DIR,
        "receptor_topology",
        "test_receptor_topology_protocol.dms",
    )
    ligand_file_name = os.path.join(
        TEST_DATA_DIR,
        "free_docking",
        "molecular_docking",
        "ligand_prepared.sdf",
    )

    receptor_builder = UnidockReceptorTopologyBuilder(
        receptor_file_name,
        prepared_hydrogen=True,
        working_dir_name=str(tmp_path),
    )
    receptor_builder.generate_receptor_topology()
    receptor_builder.analyze_receptor_topology()
    receptor_builder.get_summary_receptor_info()

    ligand_builder = UnidockLigandTopologyBuilder(
        [ligand_file_name],
        n_cpu=1,
        working_dir_name=str(tmp_path),
    )
    ligand_builder.generate_batch_ligand_topology()
    ligand_builder.get_summary_ligand_info_dict()

    engine_input = {
        "receptor": receptor_builder.atom_info_nested_list,
        **ligand_builder.summary_ligand_info_dict,
    }

    assert list(ligand_builder.summary_ligand_info_dict) == ["MOL_0"]
    assert_valid_engine_input(engine_input)


@pytest.mark.parametrize(
    "case,expected_path",
    [
        ("short_receptor_atom", "receptor[0]"),
        ("short_ligand_atom", "MOL_0.atoms[0]"),
        ("invalid_vina_type", "MOL_0.atoms[0][3]"),
        ("pair_index_out_of_range", "MOL_0.atoms[0][6][0]"),
        ("root_index_out_of_range", "MOL_0.root_atoms[0]"),
        ("short_torsion", "MOL_0.torsions[0]"),
        ("invalid_bias_r2", "MOL_0.atoms[0][8][0][4]"),
        ("non_finite_coordinate", "MOL_0.atoms[0][0]"),
        ("missing_root_atoms", "MOL_0"),
    ],
)
def test_invalid_engine_input_is_rejected(case, expected_path):
    data = minimal_engine_input()
    ligand = data["MOL_0"]

    if case == "short_receptor_atom":
        data["receptor"][0].pop()
    elif case == "short_ligand_atom":
        ligand["atoms"][0].pop()
    elif case == "invalid_vina_type":
        ligand["atoms"][0][3] = 21
    elif case == "pair_index_out_of_range":
        ligand["atoms"][0][6] = [2]
    elif case == "root_index_out_of_range":
        ligand["root_atoms"] = [2]
    elif case == "short_torsion":
        ligand["torsions"] = [
            [[0, 1, 1, 0], 0.0, [[-180.0, 180.0]], [1]]
        ]
    elif case == "invalid_bias_r2":
        ligand["atoms"][0].append([[0.0, 0.0, 0.0, -1.0, 0.0]])
    elif case == "non_finite_coordinate":
        ligand["atoms"][0][0] = float("nan")
    elif case == "missing_root_atoms":
        del ligand["root_atoms"]

    with pytest.raises(AssertionError) as exc_info:
        assert_valid_engine_input(data)

    assert expected_path in str(exc_info.value)
