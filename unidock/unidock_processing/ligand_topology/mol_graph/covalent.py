from rdkit import Chem
from unidock_processing.ligand_topology import utils
from .generic import GenericMolGraph


class CovalentMolGraph(GenericMolGraph):
    """
    CovalentMolGraph is a specialized class for handling covalent ligands in molecular graphs.
    It extends GenericMolGraph to include functionality specific to covalent interactions.
    """

    name = 'covalent'

    def __init__(
        self,
        mol:Chem.Mol,
        torsion_library_dict:dict,
        working_dir_name:str='.',
        **kwargs
    ):
        super().__init__(mol, torsion_library_dict, working_dir_name)
        self.covalent_anchor_atom_info = tuple()

    def _preprocess_mol(self):
        self.mol, self.covalent_anchor_atom_info, _ = (
            utils.prepare_covalent_ligand_mol(self.mol)
        )
        super()._preprocess_mol()

    def _get_root_atom_ids(self, splitted_mol_list:list[Chem.Mol],
                            rotatable_bond_info_list:list[tuple[int,...]]) -> list[int]:
        root_fragment_idx = None
        for fragment_idx, fragment in enumerate(splitted_mol_list):
            for atom in fragment.GetAtoms():
                atom_info = (
                    atom.GetProp('chain_idx'),
                    atom.GetProp('residue_name'),
                    atom.GetIntProp('residue_idx'),
                    atom.GetProp('atom_name'),
                )
                if atom_info == self.covalent_anchor_atom_info:
                    root_fragment_idx = fragment_idx
                    break

            if root_fragment_idx is not None:
                break

        if root_fragment_idx is None:
            raise ValueError("Bugs in root finding code for covalent docking!")

        return [atom.GetIntProp('internal_atom_idx') for atom in splitted_mol_list[root_fragment_idx].GetAtoms()]
