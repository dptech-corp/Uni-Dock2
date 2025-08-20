import json
from pathlib import Path
import os
from rdkit import Chem
from unidock_processing.torsion_library.utils import get_torsion_lib_dict
from unidock_processing.ligand_topology.mol_graph import BaseMolGraph
import pytest
from context import TEST_DATA_DIR


torsion_library_dict = get_torsion_lib_dict()


TEST_CASES_DIR = os.path.join(TEST_DATA_DIR, "ligand_topology", "align")
METHOD_LIST = ['atom_mapper_align']

def get_test_cases(case_names:list[str]):
    test_cases = []
    for case_name in case_names:
        test_case_dir = os.path.join(TEST_CASES_DIR, case_name)
        with open(os.path.join(test_case_dir, "atom_mapping.json")) as f:
            atom_mapping = {int(k): v for k, v in json.load(f).items()}
        root_atom_ids = []
        if os.path.exists(os.path.join(test_case_dir, "root_atom_ids.json")):
            with open(os.path.join(test_case_dir, "root_atom_ids.json")) as f:
                root_atom_ids = json.load(f)
        for method in METHOD_LIST:
            test_cases.append(pytest.param(
                os.path.join(test_case_dir, "query.sdf"),
                os.path.join(test_case_dir, "ref.sdf"),
                atom_mapping,
                root_atom_ids,
                method,
                id=f'{case_name}-{method}',
            ))
    return test_cases


@pytest.mark.parametrize("query_sdf_file,ref_sdf_file,atom_mapping,root_atom_ids,method",
                         get_test_cases(["check_failed_case", "root_atoms_case"]))
def test_build_mol_graph(query_sdf_file:str, ref_sdf_file:str, atom_mapping:dict, root_atom_ids:list[int],
                         method:str, tmp_path:Path):
    query_mol = Chem.SDMolSupplier(query_sdf_file, removeHs=False)[0]
    reference_mol = Chem.SDMolSupplier(ref_sdf_file, removeHs=False)[0]

    mol_graph_builder = BaseMolGraph.create(
        method,
        mol=query_mol,
        torsion_library_dict=torsion_library_dict,
        reference_mol=reference_mol,
        core_atom_mapping_dict=atom_mapping,
        working_dir_name=tmp_path,
    )
    (
        atom_info_nested_list,
        torsion_info_nested_list,
        root_atom_idx_list,
        fragment_atom_idx_nested_list,
    ) = mol_graph_builder.build_graph()

    assert root_atom_idx_list == root_atom_ids, \
        f"Root atoms mismatch: expected {root_atom_ids}, got {root_atom_idx_list}"


@pytest.mark.parametrize("query_sdf_file,ref_sdf_file,atom_mapping,root_atom_ids,method",
                         get_test_cases(["fragment_case"]))
def test_fragment_split(query_sdf_file:str, ref_sdf_file:str, atom_mapping:dict, root_atom_ids:list[int],
                        method:str, tmp_path:Path):
    query_mol = Chem.SDMolSupplier(query_sdf_file, removeHs=False)[0]
    reference_mol = Chem.SDMolSupplier(ref_sdf_file, removeHs=False)[0]

    mol_graph_builder = BaseMolGraph.create(
        method,
        mol=query_mol,
        torsion_library_dict=torsion_library_dict,
        reference_mol=reference_mol,
        core_atom_mapping_dict=atom_mapping,
        working_dir_name=tmp_path,
    )
    rot_bonds = mol_graph_builder._get_rotatable_bond_info()
    filtered_fragments = mol_graph_builder._freeze_bond(rot_bonds)
    assert len(filtered_fragments) > 1, "incorrect fragments number after freezing bonds"
