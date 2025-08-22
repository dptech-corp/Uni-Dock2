import os
from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from unidock_processing.atom_types.vina_atom_type import AtomType
from unidock_processing.atom_types.unidock_vina_atom_types import VINA_ATOM_TYPE_DICT
from unidock_processing.atom_types.unidock_ff_atom_types import FF_ATOM_TYPE_DICT
from unidock_processing.torsion_library.torsion_library_driver import TorsionLibraryDriver
from unidock_processing.ligand_topology import utils
from unidock_processing.ligand_topology.rotatable_bond import BaseRotatableBond
from .base import BaseMolGraph


class GenericMolGraph(BaseMolGraph):
    name = 'generic'

    def __init__(
        self,
        mol:Chem.Mol,
        torsion_library_dict:dict,
        working_dir_name:str='.',
        **kwargs
    ):
        self.mol = mol
        self.torsion_library_dict = torsion_library_dict
        self.working_dir_name = working_dir_name

    def _preprocess_mol(self):
        mol = self.mol
        atom_typer = AtomType()
        atom_typer.assign_atom_types(mol)
        ComputeGasteigerCharges(mol)
        utils.assign_atom_properties(mol)
        self.mol = mol

    def _get_rotatable_bond_info(self) -> list[tuple[int,...]]:
        rotatable_bond_finder = BaseRotatableBond.create('generic')
        return rotatable_bond_finder.identify_rotatable_bonds(self.mol)

    def __construct_gaff2(self) -> tuple[list[int], list[float], dict[tuple, dict]]:
        temp_ligand_sdf_file_name = os.path.join(self.working_dir_name, 'ligand.sdf')
        with Chem.SDWriter(temp_ligand_sdf_file_name) as writer:
            writer.write(self.mol)

        (
            atom_type_list,
            partial_charge_list,
            atom_parameter_dict,
            torsion_parameter_nested_dict
        ) = utils.record_gaff2_atom_types_and_parameters(
            temp_ligand_sdf_file_name, 'gas', self.working_dir_name
        )
        return atom_type_list, partial_charge_list, torsion_parameter_nested_dict

    def _freeze_bond(self, rotatable_bond_info_list:list[tuple[int,...]]) -> list[Chem.Mol]:
        mol = self.mol
        rotatable_bond_idx_list = []
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if (start_idx, end_idx) in rotatable_bond_info_list or (end_idx, start_idx) in rotatable_bond_info_list:
                rotatable_bond_idx_list.append(bond.GetIdx())

        if len(rotatable_bond_idx_list) != 0:
            splitted_mol = FragmentOnBonds(mol, rotatable_bond_idx_list, addDummies=False)
            splitted_mol_list = list(GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False))
        else:
            splitted_mol_list = [mol]

        return splitted_mol_list

    def _get_root_atom_ids(self, splitted_mol_list:list[Chem.Mol],
                          rotatable_bond_info_list:list[tuple[int,...]]) -> list[int]:
        root_fragment_idx = utils.root_finding_strategy(splitted_mol_list, rotatable_bond_info_list)
        return [atom.GetIntProp('internal_atom_idx') for atom in splitted_mol_list[root_fragment_idx].GetAtoms()]

    def __get_frags_atom_ids(self, splitted_mol_list:list[Chem.Mol]) -> list[list[int]]:
        return [[atom.GetIntProp('internal_atom_idx') for atom in fragment_mol.GetAtoms()] \
                for fragment_mol in splitted_mol_list]

    def __get_atoms_info(self, atom_type_list:list[str],
                      partial_charge_list:list[float],
                      atom_pair_12_13_nested_list:list[list[int]],
                      atom_pair_14_nested_list:list[list[int]]
        ) -> list[tuple[float, float, float, int, int, float, list[int], list[int]]]:
        mol = self.mol
        atom_info_nested_list = []
        for atom_idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(atom_idx)
            ff_atom_type = atom_type_list[atom_idx]
            vina_atom_type = atom.GetProp('vina_atom_type')

            atom_info = (
                atom.GetDoubleProp('x'),
                atom.GetDoubleProp('y'),
                atom.GetDoubleProp('z'),
                VINA_ATOM_TYPE_DICT[vina_atom_type],
                FF_ATOM_TYPE_DICT[ff_atom_type],
                partial_charge_list[atom_idx],
                atom_pair_12_13_nested_list[atom_idx],
                atom_pair_14_nested_list[atom_idx]
            )

            atom_info_nested_list.append(atom_info)

        return atom_info_nested_list

    def __get_torsions_info(self,
                          rotatable_bond_info_list:list[tuple[int,...]],
                          root_atom_idx_first:int,
                          atom_type_list:list[int],
                          torsion_parameter_nested_dict,
        ):
        torsion_library_driver = TorsionLibraryDriver(self.mol, rotatable_bond_info_list, self.torsion_library_dict)
        torsion_library_driver.perform_torsion_matches()
        torsion_library_driver.identify_torsion_mobile_atoms(root_atom_idx_first)

        torsion_info_nested_list = [None] * torsion_library_driver.num_rotatable_bonds
        for torsion_idx in range(torsion_library_driver.num_rotatable_bonds):

            torsion_atom_idx_list = torsion_library_driver.torsion_atom_idx_nested_list[torsion_idx]
            torsion_value_list = torsion_library_driver.original_torsion_value_list[torsion_idx]
            torsion_range_list = torsion_library_driver.enumerated_torsion_range_nested_list[torsion_idx]
            torsion_mobile_atom_idx_list = torsion_library_driver.mobile_atom_idx_nested_list[torsion_idx]

            torsion_type = [atom_type_list[torsion_atom_idx_list[i]] for i in range(4)]
            if tuple(torsion_type) not in torsion_parameter_nested_dict:
                torsion_type = reversed(torsion_type)
            torsion_parameter_dict_list = torsion_parameter_nested_dict[tuple(torsion_type)]

            torsion_parameter_nested_list = [
                [
                    torsion_parameter_dict['barrier_factor'],
                    torsion_parameter_dict['barrier_height'],
                    torsion_parameter_dict['periodicity'],
                    torsion_parameter_dict['phase']
                ] for torsion_parameter_dict in torsion_parameter_dict_list
            ]

            torsion_info_list = [
                torsion_atom_idx_list,
                torsion_value_list,
                torsion_range_list,
                torsion_mobile_atom_idx_list,
                torsion_parameter_nested_list
            ]

            torsion_info_nested_list[torsion_idx] = torsion_info_list

        return torsion_info_nested_list

    def build_graph(self):
        """
        Build the molecular graph representation.

        This method constructs the graph structure from the RDKit molecule,
        including atoms and bonds.
        """
        self._preprocess_mol()

        rotatable_bond_info_list = self._get_rotatable_bond_info()

        atom_pair_12_13_nested_list, atom_pair_14_nested_list = utils.calculate_nonbonded_atom_pairs(self.mol)

        atom_type_list, partial_charge_list, torsion_parameter_nested_dict = self.__construct_gaff2()

        splitted_mol_list = self._freeze_bond(rotatable_bond_info_list)

        root_atom_idx_list = self._get_root_atom_ids(splitted_mol_list, rotatable_bond_info_list)

        fragment_atom_idx_nested_list = self.__get_frags_atom_ids(splitted_mol_list)

        atom_info_nested_list = self.__get_atoms_info(atom_type_list, partial_charge_list,
                                                      atom_pair_12_13_nested_list,
                                                      atom_pair_14_nested_list)

        torsion_info_nested_list = self.__get_torsions_info(rotatable_bond_info_list, root_atom_idx_list[0],
                                                            atom_type_list, torsion_parameter_nested_dict)

        return (
            atom_info_nested_list,
            torsion_info_nested_list,
            root_atom_idx_list,
            fragment_atom_idx_nested_list,
        )
