import os
import re
from rdkit import Chem
from unidock_processing.ligand_topology import utils
from unidock_processing.ligand_topology.rotatable_bond import BaseRotatableBond
from .generic import GenericMolGraph


class TemplateMolGraph(GenericMolGraph):
    """
    A class for constructing molecular graph for template docking.
    """

    name = 'template'

    def __init__(
        self,
        mol:Chem.Mol,
        torsion_library_dict:dict,
        reference_mol:Chem.Mol=None,
        core_atom_mapping_dict:dict=None,
        construct_ff=False,
        working_dir_name:str='.',
    ):
        super().__init__(mol, torsion_library_dict, construct_ff, working_dir_name)
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
                raise ValueError("Specified core atom mapping makes unconnected fragments!!")

        core_atom_idx_list = utils.get_core_alignment_for_template_docking(
            reference_mol, mol, core_atom_mapping_dict
        )

        ## The tail parts of atoms in the core are cancelled and not beloings to the core itself. Currently this strategy is disabled temporarily. #noqa
        # for core_atom_idx in core_atom_idx_list:
        #     core_atom = mol.GetAtomWithIdx(core_atom_idx)
        #     for neighbor_atom in core_atom.GetNeighbors():
        #         if neighbor_atom.GetIdx() not in core_atom_idx_list:
        #             core_atom_idx_list.remove(core_atom_idx)
        #             break

        with Chem.SDWriter(os.path.join(self.working_dir_name, 'ligand_template_aligned.sdf')) as writer:
            writer.write(mol)

        super().preprocess_mol()
        self.core_atom_idx_list = core_atom_idx_list

    def get_rotatable_bond_info(self) -> list[tuple[int,...]]:
        rotatable_bond_finder = BaseRotatableBond.create('generic')
        rotatable_bond_info_list = rotatable_bond_finder.identify_rotatable_bonds(self.mol)
        rotatable_bond_info_list = [bond_info for bond_info in rotatable_bond_info_list \
                                    if bond_info[0] in self.core_atom_idx_list \
                                        and bond_info[1] in self.core_atom_idx_list]
        return rotatable_bond_info_list

    def get_root_atom_ids(self, splitted_mol_list:list[Chem.Mol],
                            rotatable_bond_info_list:list[tuple[int,...]]) -> list[int]:
        root_fragment_idx = None
        for fragment_idx, fragment in enumerate(splitted_mol_list):
            for atom in fragment.GetAtoms():
                internal_atom_idx = int(re.split(r"(\d+)", atom.GetProp('atom_name'))[1]) - 1
                if internal_atom_idx in self.core_atom_idx_list:
                    root_fragment_idx = fragment_idx
                    break

            if root_fragment_idx is not None:
                break

        if root_fragment_idx is None:
            raise ValueError("Bugs in root finding code for template docking!")

        return [atom.GetIntProp('internal_atom_idx') for atom in splitted_mol_list[root_fragment_idx].GetAtoms()]
