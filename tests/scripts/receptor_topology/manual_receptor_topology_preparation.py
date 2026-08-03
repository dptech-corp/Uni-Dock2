import os
import pytest

from context import TEST_DATA_DIR

from unidock2.unidocktools.receptor_topology_preparation import (
    ReceptorTopologyPreparation,
)

# Atom counts captured from the MDAnalysis-based implementation so that the
# OpenMM-based one stays byte-for-byte compatible with it.
RECEPTOR_CASE_LIST = [
    (
        os.path.join(
            TEST_DATA_DIR, 'receptor_topology', 'test_receptor_topology_protocol.pdb'
        ),
        4667,
        4660,
    ),
    (
        os.path.join(
            TEST_DATA_DIR, 'receptor_topology', 'test_receptor_topology_RNA.pdb'
        ),
        3484,
        3485,
    ),
    (
        os.path.join(
            TEST_DATA_DIR,
            'free_docking',
            'virtual_screening',
            '5WIU_protein_cleaned.pdb',
        ),
        2917,
        2918,
    ),
]

RECEPTOR_CASE_ID_LIST = [
    os.path.basename(receptor_pdb_file_name)
    for receptor_pdb_file_name, _, _ in RECEPTOR_CASE_LIST
]

def parse_pdb_atom_record_list(pdb_file_name):
    atom_record_list = []
    with open(pdb_file_name) as pdb_file:
        for pdb_line in pdb_file:
            if not pdb_line.startswith(('ATOM  ', 'HETATM')):
                continue

            atom_record_list.append(
                {
                    'atom_name': pdb_line[12:16].strip(),
                    'residue_name': pdb_line[17:20].strip(),
                    'chain_idx': pdb_line[21],
                    'residue_idx': pdb_line[22:27].strip(),
                    'x': round(float(pdb_line[30:38]), 3),
                    'y': round(float(pdb_line[38:46]), 3),
                    'z': round(float(pdb_line[46:54]), 3),
                }
            )

    return atom_record_list

def get_atom_coordinate_set(atom_record_list):
    return {
        (
            atom_record['atom_name'],
            atom_record['residue_name'],
            atom_record['x'],
            atom_record['y'],
            atom_record['z'],
        )
        for atom_record in atom_record_list
    }

@pytest.fixture(scope='module')
def prepared_receptor_dict(tmp_path_factory):
    """Run the receptor preparation once per input structure."""

    prepared_receptor_dict = {}
    for receptor_pdb_file_name, _, _ in RECEPTOR_CASE_LIST:
        working_dir_name = tmp_path_factory.mktemp(
            os.path.splitext(os.path.basename(receptor_pdb_file_name))[0]
        )

        receptor_topology_preparation = ReceptorTopologyPreparation(
            receptor_pdb_file_name, str(working_dir_name)
        )
        receptor_topology_preparation.run_preparation()

        prepared_receptor_dict[receptor_pdb_file_name] = receptor_topology_preparation

    return prepared_receptor_dict

@pytest.mark.parametrize(
    'receptor_pdb_file_name, num_cleaned_atoms, num_fixed_atoms',
    RECEPTOR_CASE_LIST,
    ids=RECEPTOR_CASE_ID_LIST,
)
def test_cleaned_structure_drops_hydrogen_and_oxt_atoms(
    prepared_receptor_dict, receptor_pdb_file_name, num_cleaned_atoms, num_fixed_atoms
):
    receptor_topology_preparation = prepared_receptor_dict[receptor_pdb_file_name]
    cleaned_atom_record_list = parse_pdb_atom_record_list(
        receptor_topology_preparation.receptor_cleaned_pdb_file_name
    )

    assert len(cleaned_atom_record_list) == num_cleaned_atoms

    for atom_record in cleaned_atom_record_list:
        assert atom_record['atom_name'] != 'OXT'
        assert not atom_record['atom_name'].startswith('H')

@pytest.mark.parametrize(
    'receptor_pdb_file_name, num_cleaned_atoms, num_fixed_atoms',
    RECEPTOR_CASE_LIST,
    ids=RECEPTOR_CASE_ID_LIST,
)
def test_cleaned_structure_preserves_input_coordinates(
    prepared_receptor_dict, receptor_pdb_file_name, num_cleaned_atoms, num_fixed_atoms
):
    receptor_topology_preparation = prepared_receptor_dict[receptor_pdb_file_name]
    input_coordinate_set = get_atom_coordinate_set(
        parse_pdb_atom_record_list(receptor_pdb_file_name)
    )
    cleaned_coordinate_set = get_atom_coordinate_set(
        parse_pdb_atom_record_list(
            receptor_topology_preparation.receptor_cleaned_pdb_file_name
        )
    )

    assert cleaned_coordinate_set.issubset(input_coordinate_set)

@pytest.mark.parametrize(
    'receptor_pdb_file_name, num_cleaned_atoms, num_fixed_atoms',
    RECEPTOR_CASE_LIST,
    ids=RECEPTOR_CASE_ID_LIST,
)
def test_cleaned_structure_keeps_one_alternate_location(
    prepared_receptor_dict, receptor_pdb_file_name, num_cleaned_atoms, num_fixed_atoms
):
    receptor_topology_preparation = prepared_receptor_dict[receptor_pdb_file_name]
    cleaned_atom_record_list = parse_pdb_atom_record_list(
        receptor_topology_preparation.receptor_cleaned_pdb_file_name
    )

    atom_key_list = [
        (
            atom_record['chain_idx'],
            atom_record['residue_idx'],
            atom_record['atom_name'],
        )
        for atom_record in cleaned_atom_record_list
    ]

    assert len(atom_key_list) == len(set(atom_key_list))

@pytest.mark.parametrize(
    'receptor_pdb_file_name, num_cleaned_atoms, num_fixed_atoms',
    RECEPTOR_CASE_LIST,
    ids=RECEPTOR_CASE_ID_LIST,
)
def test_final_structure_only_renames_cysteine_residues(
    prepared_receptor_dict, receptor_pdb_file_name, num_cleaned_atoms, num_fixed_atoms
):
    receptor_topology_preparation = prepared_receptor_dict[receptor_pdb_file_name]
    fixed_atom_record_list = parse_pdb_atom_record_list(
        receptor_topology_preparation.receptor_fixed_pdb_file_name
    )
    final_atom_record_list = parse_pdb_atom_record_list(
        receptor_topology_preparation.receptor_final_pdb_file_name
    )

    assert len(fixed_atom_record_list) == num_fixed_atoms
    assert len(final_atom_record_list) == num_fixed_atoms

    for fixed_atom_record, final_atom_record in zip(
        fixed_atom_record_list, final_atom_record_list
    ):
        expected_residue_name = fixed_atom_record['residue_name']
        if expected_residue_name == 'CYS':
            expected_residue_name = 'CYX'

        assert final_atom_record['residue_name'] == expected_residue_name
        assert final_atom_record['atom_name'] == fixed_atom_record['atom_name']
        assert final_atom_record['chain_idx'] == fixed_atom_record['chain_idx']
        assert final_atom_record['residue_idx'] == fixed_atom_record['residue_idx']
        assert final_atom_record['x'] == fixed_atom_record['x']
        assert final_atom_record['y'] == fixed_atom_record['y']
        assert final_atom_record['z'] == fixed_atom_record['z']

    assert all(
        atom_record['residue_name'] != 'CYS' for atom_record in final_atom_record_list
    )

@pytest.mark.parametrize(
    'receptor_pdb_file_name, num_cleaned_atoms, num_fixed_atoms',
    RECEPTOR_CASE_LIST,
    ids=RECEPTOR_CASE_ID_LIST,
)
def test_parameterized_dms_file_is_generated(
    prepared_receptor_dict, receptor_pdb_file_name, num_cleaned_atoms, num_fixed_atoms
):
    receptor_topology_preparation = prepared_receptor_dict[receptor_pdb_file_name]

    assert os.path.isfile(receptor_topology_preparation.receptor_dms_file_name)
    assert os.path.getsize(receptor_topology_preparation.receptor_dms_file_name) > 0
