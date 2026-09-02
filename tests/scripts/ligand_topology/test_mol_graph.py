import json
from pathlib import Path
import os
from rdkit import Chem
from unidock2.torsion_library.utils import get_torsion_lib_dict
from unidock2.ligand_topology.mol_graph import BaseMolGraph
from unidock2.ligand_topology.mol_graph.generic import GenericMolGraph
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
        construct_ff=False,
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
        construct_ff=False,
        working_dir_name=tmp_path,
    )
    rot_bonds = mol_graph_builder.get_rotatable_bond_info()
    filtered_fragments = mol_graph_builder.freeze_bond(rot_bonds)
    assert len(filtered_fragments) > 1, "incorrect fragments number after freezing bonds"


@pytest.fixture
def torsion_case_torsion_result() -> set[tuple[int]]:
    return set([(12, 13, 14, 15), (13, 14, 15, 16)])


@pytest.mark.parametrize("query_sdf_file,ref_sdf_file,atom_mapping,root_atom_ids,method",
                         get_test_cases(["torsion_case"]))
def test_torsion(query_sdf_file:str, ref_sdf_file:str, atom_mapping:dict, root_atom_ids:list[int],
                 method:str, tmp_path:Path, torsion_case_torsion_result:list[tuple[int]]):
    query_mol = Chem.SDMolSupplier(query_sdf_file, removeHs=False)[0]
    reference_mol = Chem.SDMolSupplier(ref_sdf_file, removeHs=False)[0]

    mol_graph_builder = BaseMolGraph.create(
        method,
        mol=query_mol,
        torsion_library_dict=torsion_library_dict,
        reference_mol=reference_mol,
        core_atom_mapping_dict=atom_mapping,
        construct_ff=False,
        working_dir_name=tmp_path,
    )
    (
        atom_info_nested_list,
        torsion_info_nested_list,
        root_atom_idx_list,
        fragment_atom_idx_nested_list,
    ) = mol_graph_builder.build_graph()
    torsion_ids = set([tuple(t[0]) for t in torsion_info_nested_list])

    assert torsion_ids == torsion_case_torsion_result, \
        f"Torsion info mismatch: expected {torsion_case_torsion_result}, got {torsion_ids}"


def test_force_field_construction_leaves_engine_read_fields_untouched(tmp_path, monkeypatch):
    ligand_sdf_file = os.path.join(
        TEST_DATA_DIR,
        "free_docking",
        "molecular_docking",
        "ligand_prepared.sdf",
    )

    def build_graph(construct_ff):
        mol = Chem.SDMolSupplier(ligand_sdf_file, removeHs=False)[0]
        graph = GenericMolGraph(
            mol=mol,
            torsion_library_dict=torsion_library_dict,
            construct_ff=construct_ff,
            working_dir_name=str(tmp_path),
        )
        return graph.build_graph()

    disabled_graph = build_graph(False)
    calls = []
    torsion_parameter = {
        "barrier_factor": 1,
        "barrier_height": 2.0,
        "periodicity": 3,
        "phase": 180.0,
    }

    def fake_construct_gaff2(self):
        calls.append(self)
        num_atoms = self.mol.GetNumAtoms()
        return (
            ["c"] * num_atoms,
            [float(atom_idx) for atom_idx in range(num_atoms)],
            {("c", "c", "c", "c"): [torsion_parameter]},
        )

    monkeypatch.setattr(GenericMolGraph, "construct_gaff2", fake_construct_gaff2)
    enabled_graph = build_graph(True)

    disabled_atoms, disabled_torsions, disabled_root, disabled_fragments = disabled_graph
    enabled_atoms, enabled_torsions, enabled_root, enabled_fragments = enabled_graph

    assert len(calls) == 1
    assert disabled_root == enabled_root
    assert disabled_fragments == enabled_fragments
    assert len(disabled_atoms) == len(enabled_atoms)
    assert len(disabled_torsions) == len(enabled_torsions)
    # Slots 4 and 5 carry force-field type and charge, which the engine skips.
    # Everything the engine does read must be identical either way.
    for disabled_atom, enabled_atom in zip(disabled_atoms, enabled_atoms):
        assert disabled_atom[:4] == enabled_atom[:4]
        assert disabled_atom[6:] == enabled_atom[6:]
    for disabled_torsion, enabled_torsion in zip(
        disabled_torsions, enabled_torsions
    ):
        assert disabled_torsion[:4] == enabled_torsion[:4]
        assert disabled_torsion[4] == []
        assert enabled_torsion[4] == [[1, 2.0, 3, 180.0]]
