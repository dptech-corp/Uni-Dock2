import os
import shutil
import uuid
import pytest
import yaml

from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from unidock.unidock_processing.ligand_topology import utils
from unidock.unidock_processing.ligand_topology.generic_rotatable_bond import GenericRotatableBond

from unidock.unidock_processing.unidocktools.vina_atom_type import AtomType
from unidock.unidock_processing.unidocktools.unidock_vina_atom_types import VINA_ATOM_TYPE_DICT

@pytest.fixture
def template_configurations():
    return os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'unidock_configurations.yaml')

@pytest.fixture
def nonbonded_exclusion_test_molecule():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ligand_topology')
    ligand_sdf_file_name = os.path.join(data_file_dir_name, 'test_nonbonded_exclusion.sdf')

    return ligand_sdf_file_name

@pytest.fixture
def root_finding_test_molecule():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ligand_topology')
    ligand_sdf_file_name = os.path.join(data_file_dir_name, 'test_root_finding_8E77_ligand_prepared.sdf')

    return ligand_sdf_file_name

@pytest.fixture
def vina_atom_type_test_molecule_1():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ligand_topology')
    ligand_sdf_file_name = os.path.join(data_file_dir_name, 'test_vina_atom_type_1.sdf')

    return ligand_sdf_file_name

@pytest.fixture
def vina_atom_type_test_molecule_2():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ligand_topology')
    ligand_sdf_file_name = os.path.join(data_file_dir_name, 'test_vina_atom_type_2.sdf')

    return ligand_sdf_file_name

@pytest.fixture
def vina_atom_type_test_molecule_3():
    data_file_dir_name = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ligand_topology')
    ligand_sdf_file_name = os.path.join(data_file_dir_name, 'test_vina_atom_type_3.sdf')

    return ligand_sdf_file_name

def test_nonbonded_exclusion(nonbonded_exclusion_test_molecule):
    mol = Chem.SDMolSupplier(nonbonded_exclusion_test_molecule, removeHs=False)[0]
    atom_pair_12_13_nested_list, atom_pair_14_nested_list = utils.calculate_nonbonded_atom_pairs(mol)

    valid_atom_pair_12_13_nested_list = [[1, 2, 3, 10, 11, 23, 24, 30],
                                         [0, 2, 3, 4, 7, 10, 24],
                                         [0, 1, 3],
                                         [0, 1, 2, 4, 5, 6, 7, 8],
                                         [1, 3, 5, 6, 7],
                                         [3, 4, 6, 7],
                                         [3, 4, 5, 7, 8],
                                         [1, 3, 4, 5, 6, 8, 9, 25, 26],
                                         [3, 6, 7, 9, 25, 26, 27, 28, 29],
                                         [7, 8, 25, 26, 27, 28, 29],
                                         [0, 1, 11, 12, 22, 23, 24, 30, 31, 52],
                                         [0, 10, 12, 13, 22, 23, 30, 31, 32, 33, 52],
                                         [10, 11, 13, 14, 22, 23, 31, 32, 33, 34],
                                         [11, 12, 14, 15, 22, 23, 32, 33, 34, 35, 36, 50, 51],
                                         [12, 13, 15, 16, 21, 22, 34, 35, 36, 37],
                                         [13, 14, 16, 17, 20, 21, 35, 36, 37, 38, 39, 48, 49],
                                         [14, 15, 17, 18, 21, 37, 38, 39, 40, 41],
                                         [15, 16, 18, 19, 38, 39, 40, 41, 42, 43],
                                         [16, 17, 19, 20, 40, 41, 42, 43, 44, 45],
                                         [17, 18, 20, 21, 42, 43, 44, 45, 46, 47],
                                         [15, 18, 19, 21, 44, 45, 46, 47, 48, 49],
                                         [14, 15, 16, 19, 20, 37, 46, 47, 48, 49],
                                         [10, 11, 12, 13, 14, 23, 34, 50, 51, 52],
                                         [0, 10, 11, 12, 13, 22, 30, 31, 50, 51, 52],
                                         [0, 1, 10],
                                         [7, 8, 9, 26],
                                         [7, 8, 9, 25],
                                         [8, 9, 28, 29],
                                         [8, 9, 27, 29],
                                         [8, 9, 27, 28],
                                         [0, 10, 11, 23],
                                         [10, 11, 12, 23],
                                         [11, 12, 13, 33],
                                         [11, 12, 13, 32],
                                         [12, 13, 14, 22],
                                         [13, 14, 15, 36],
                                         [13, 14, 15, 35],
                                         [14, 15, 16, 21],
                                         [15, 16, 17, 39],
                                         [15, 16, 17, 38],
                                         [16, 17, 18, 41],
                                         [16, 17, 18, 40],
                                         [17, 18, 19, 43],
                                         [17, 18, 19, 42],
                                         [18, 19, 20, 45],
                                         [18, 19, 20, 44],
                                         [19, 20, 21, 47],
                                         [19, 20, 21, 46],
                                         [15, 20, 21, 49],
                                         [15, 20, 21, 48],
                                         [13, 22, 23, 51],
                                         [13, 22, 23, 50],
                                         [10, 11, 22, 23]]

    valid_atom_pair_14_nested_list = [[4, 7, 12, 22, 31, 52],
                                      [5, 6, 8, 11, 23, 30],
                                      [4, 7, 10, 24],
                                      [9, 10, 24, 25, 26],
                                      [0, 2, 8],
                                      [1, 8],
                                      [1, 9, 25, 26],
                                      [0, 2, 27, 28, 29],
                                      [1, 4, 5],
                                      [3, 6],
                                      [2, 3, 13, 32, 33, 50, 51],
                                      [1, 14, 24, 34, 50, 51],
                                      [0, 15, 30, 35, 36, 50, 51, 52],
                                      [10, 16, 21, 31, 37, 52],
                                      [11, 17, 20, 23, 32, 33, 38, 39, 48, 49, 50, 51],
                                      [12, 18, 19, 22, 34, 40, 41, 46, 47],
                                      [13, 19, 20, 35, 36, 42, 43, 48, 49],
                                      [14, 20, 21, 37, 44, 45],
                                      [15, 21, 38, 39, 46, 47],
                                      [15, 16, 40, 41, 48, 49],
                                      [14, 16, 17, 37, 42, 43],
                                      [13, 17, 18, 35, 36, 38, 39, 44, 45],
                                      [0, 15, 30, 31, 32, 33, 35, 36],
                                      [1, 14, 24, 32, 33, 34],
                                      [2, 3, 11, 23, 30],
                                      [3, 6, 27, 28, 29],
                                      [3, 6, 27, 28, 29],
                                      [7, 25, 26],
                                      [7, 25, 26],
                                      [7, 25, 26],
                                      [1, 12, 22, 24, 31, 52],
                                      [0, 13, 22, 30, 32, 33, 52],
                                      [10, 14, 22, 23, 31, 34],
                                      [10, 14, 22, 23, 31, 34],
                                      [11, 15, 23, 32, 33, 35, 36, 50, 51],
                                      [12, 16, 21, 22, 34, 37],
                                      [12, 16, 21, 22, 34, 37],
                                      [13, 17, 20, 35, 36, 38, 39, 48, 49],
                                      [14, 18, 21, 37, 40, 41],
                                      [14, 18, 21, 37, 40, 41],
                                      [15, 19, 38, 39, 42, 43],
                                      [15, 19, 38, 39, 42, 43],
                                      [16, 20, 40, 41, 44, 45],
                                      [16, 20, 40, 41, 44, 45],
                                      [17, 21, 42, 43, 46, 47],
                                      [17, 21, 42, 43, 46, 47],
                                      [15, 18, 44, 45, 48, 49],
                                      [15, 18, 44, 45, 48, 49],
                                      [14, 16, 19, 37, 46, 47],
                                      [14, 16, 19, 37, 46, 47],
                                      [10, 11, 12, 14, 34, 52],
                                      [10, 11, 12, 14, 34, 52],
                                      [0, 12, 13, 30, 31, 50, 51]]

    assert atom_pair_12_13_nested_list == valid_atom_pair_12_13_nested_list
    assert atom_pair_14_nested_list == valid_atom_pair_14_nested_list

def test_root_finding_strategy(root_finding_test_molecule):
    mol = Chem.SDMolSupplier(root_finding_test_molecule, removeHs=False)[0]

    ComputeGasteigerCharges(mol)
    utils.assign_atom_properties(mol)

    rotatable_bond_finder = GenericRotatableBond()
    rotatable_bond_info_list = rotatable_bond_finder.identify_rotatable_bonds(mol)

    bond_list = list(mol.GetBonds())
    rotatable_bond_idx_list = []
    for bond in bond_list:
        bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
        if bond_info in rotatable_bond_info_list or bond_info_reversed in rotatable_bond_info_list:
            rotatable_bond_idx_list.append(bond.GetIdx())

    if len(rotatable_bond_idx_list) != 0:
        splitted_mol = FragmentOnBonds(mol, rotatable_bond_idx_list, addDummies=False)
        splitted_mol_list = list(GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False))
    else:
        splitted_mol_list = [mol]

    root_fragment_idx = utils.root_finding_strategy(splitted_mol_list, rotatable_bond_info_list)
    valid_root_fragment_idx = 5

    assert root_fragment_idx == valid_root_fragment_idx

def test_vina_atom_typing(vina_atom_type_test_molecule_1,
                          vina_atom_type_test_molecule_2,
                          vina_atom_type_test_molecule_3):

    atom_typer = AtomType()
    mol_1 = Chem.SDMolSupplier(vina_atom_type_test_molecule_1, removeHs=False)[0]
    mol_2 = Chem.SDMolSupplier(vina_atom_type_test_molecule_2, removeHs=False)[0]
    mol_3 = Chem.SDMolSupplier(vina_atom_type_test_molecule_3, removeHs=False)[0]

    atom_typer = AtomType()
    atom_typer.assign_atom_types(mol_1)
    atom_typer.assign_atom_types(mol_2)
    atom_typer.assign_atom_types(mol_3)

    num_atoms_1 = mol_1.GetNumAtoms()
    atom_type_list_1 = [None] * num_atoms_1
    for atom_idx in range(num_atoms_1):
        atom = mol_1.GetAtomWithIdx(atom_idx)
        atom_type = VINA_ATOM_TYPE_DICT[atom.GetProp('vina_atom_type')]
        atom_type_list_1[atom_idx] = atom_type

    num_atoms_2 = mol_2.GetNumAtoms()
    atom_type_list_2 = [None] * num_atoms_2
    for atom_idx in range(num_atoms_2):
        atom = mol_2.GetAtomWithIdx(atom_idx)
        atom_type = VINA_ATOM_TYPE_DICT[atom.GetProp('vina_atom_type')]
        atom_type_list_2[atom_idx] = atom_type

    num_atoms_3 = mol_3.GetNumAtoms()
    atom_type_list_3 = [None] * num_atoms_3
    for atom_idx in range(num_atoms_3):
        atom = mol_3.GetAtomWithIdx(atom_idx)
        atom_type = VINA_ATOM_TYPE_DICT[atom.GetProp('vina_atom_type')]
        atom_type_list_3[atom_idx] = atom_type

    valid_atom_type_list_1 = [7, 3, 3, 5, 2, 2, 2, 2, 5, 2, 2, 2, 6, 3, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    valid_atom_type_list_2 = [2, 3, 10, 2, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 3, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 10, 5, 6, 2,
                              16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    valid_atom_type_list_3 = [6, 3, 2, 5, 3, 10, 2, 2, 2, 2, 2, 3, 3, 4, 3, 15, 3, 2, 2, 3, 10, 5, 3, 2, 2, 3, 10, 10,
                              3, 10, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    assert atom_type_list_1 == valid_atom_type_list_1
    assert atom_type_list_2 == valid_atom_type_list_2
    assert atom_type_list_3 == valid_atom_type_list_3
