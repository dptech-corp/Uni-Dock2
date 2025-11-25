import re
from rdkit import Chem
from unidock_processing.ligand_topology import utils
from unidock_processing.ligand_topology.rotatable_bond import BaseRotatableBond
from .generic import GenericMolGraph

class AlignMolGraph(GenericMolGraph):
    """
    A class for constructing molecular graph for alignment docking.
    """

    name = 'atom_mapper_align'

    def __init__(
        self,
        mol:Chem.Mol,
        torsion_library_dict:dict,
        reference_mol:Chem.Mol=None,
        core_atom_mapping_dict:dict=None,
        construct_ff=False,
        working_dir_name:str='.',
    ):
        super().__init__(mol, torsion_library_dict, working_dir_name)
        self.reference_mol = reference_mol
        self.core_atom_mapping_dict = core_atom_mapping_dict
        self.core_atom_idx_list = []

    def preprocess_mol(self):
        mol = self.mol
        reference_mol = self.reference_mol
        core_atom_mapping_dict = self.core_atom_mapping_dict
        if core_atom_mapping_dict is None:
            core_atom_mapping_dict = utils.get_template_docking_atom_mapping(reference_mol, mol)
        else:
            if not utils.check_manual_atom_mapping_connection(reference_mol, mol, core_atom_mapping_dict):
                print("Specified core atom mapping makes unconnected fragments!!")

        core_atom_idx_list = utils.get_full_ring_core_atoms(mol, list(core_atom_mapping_dict.values()))

        super().preprocess_mol()
        self.core_atom_idx_list = core_atom_idx_list

    def get_rotatable_bond_info(self) -> list[tuple[int,...]]:
        mol = self.mol
        core_atom_idx_list = self.core_atom_idx_list
        rotatable_bond_finder = BaseRotatableBond.create('atom_mapper_align')
        rotatable_bond_info_list = rotatable_bond_finder.identify_rotatable_bonds(mol)

        filtered_rotatable_bond_info_list = []
        for rotatable_bond_info in rotatable_bond_info_list:
            rotatable_begin_atom_idx = rotatable_bond_info[0]
            rotatable_end_atom_idx = rotatable_bond_info[1]
            if (
                rotatable_begin_atom_idx in core_atom_idx_list
                and rotatable_end_atom_idx in core_atom_idx_list
            ):
                begin_neighbors = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(rotatable_begin_atom_idx).GetNeighbors()
                                   if nbr.GetIdx() != rotatable_end_atom_idx and nbr.GetAtomicNum() > 1]
                end_neighbors = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(rotatable_end_atom_idx).GetNeighbors()
                                   if nbr.GetIdx() != rotatable_begin_atom_idx and nbr.GetAtomicNum() > 1]
                if not begin_neighbors or not end_neighbors:
                    continue

                # Check all possible torsion combinations
                has_fully_mapped_torsion = False
                for begin_neighbor in begin_neighbors:
                    for end_neighbor in end_neighbors:
                        # Form torsion: begin_neighbor - begin_idx - end_idx - end_neighbor
                        torsion_atoms = {
                            begin_neighbor, rotatable_begin_atom_idx,
                            rotatable_end_atom_idx, end_neighbor
                        }
                        mapped_torsion_atoms = len(torsion_atoms.intersection(core_atom_idx_list))

                        # If this torsion has all 4 atoms mapped, we found a fully constrained torsion
                        if mapped_torsion_atoms == 4:
                            has_fully_mapped_torsion = True
                            break
                    if has_fully_mapped_torsion:
                        break

                # If no torsion combination has all 4 atoms mapped, skip this rotatable bond
                if not has_fully_mapped_torsion:
                    filtered_rotatable_bond_info_list.append(rotatable_bond_info)
                    continue
            else:
                filtered_rotatable_bond_info_list.append(rotatable_bond_info)

        return filtered_rotatable_bond_info_list

    def get_root_atom_ids(self, splitted_mol_list:list[Chem.Mol],
                            rotatable_bond_info_list:list[tuple[int,...]]) -> list[int]:
        core_atom_idx_list = self.core_atom_idx_list
        root_fragment_idx = None
        for fragment_idx, fragment in enumerate(splitted_mol_list):
            for atom in fragment.GetAtoms():
                internal_atom_idx = int(re.split(r"(\d+)", atom.GetProp('atom_name'))[1]) - 1
                if internal_atom_idx in core_atom_idx_list:
                    nbr_atoms_heavy = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]
                    if len(nbr_atoms_heavy) > 0:
                        nbr_heavy_atom_idx_list = []
                        for nbr_atom in nbr_atoms_heavy:
                            nbr_heavy_atom_idx = int(re.split(r"(\d+)", nbr_atom.GetProp("atom_name"))[1]) - 1
                            nbr_heavy_atom_idx_list.append(nbr_heavy_atom_idx)
                        # if one of the nbr_heavy_atom_idx is in core_atom_idx_list, then this atom is the root atom
                        if any(nbr_heavy_atom_idx in core_atom_idx_list \
                               for nbr_heavy_atom_idx in nbr_heavy_atom_idx_list):
                            root_fragment_idx = fragment_idx
                            break

            if root_fragment_idx is not None:
                break

        if root_fragment_idx is None:
            raise ValueError("Bugs in root finding code for align docking!")

        return [atom.GetIntProp('internal_atom_idx') for atom in splitted_mol_list[root_fragment_idx].GetAtoms()]
