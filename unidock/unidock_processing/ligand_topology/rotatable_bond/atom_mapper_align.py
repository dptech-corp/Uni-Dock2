from rdkit import Chem
from .base import BaseRotatableBond


class AlignRotatableBond(BaseRotatableBond):
    name = 'atom_mapper_align'

    def __init__(self):
        self.rotatable_bond_smarts = (
            "[!$(*#*)&!D1&!$([CH3])&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$"
            "(C([CH3])([CH3])[CH3])]-!@[!$(*#*)&!D1&!$([CH3])&!$(C(F)(F)F)&!$"
            "(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
        )
        self.amide_bond_smarts = "[C&$(C=O)]-[N&$(NC=O);v3;H1,H2;0]"


        self.rotatable_bond_pattern = Chem.MolFromSmarts(self.rotatable_bond_smarts)
        self.amide_bond_pattern = Chem.MolFromSmarts(self.amide_bond_smarts)


    def identify_rotatable_bonds(self, mol:Chem.Mol):
        default_rotatable_bond_info_list = list(
            mol.GetSubstructMatches(self.rotatable_bond_pattern)
        )

        exclude_rotatable_bond_info_list = []
        exclude_rotatable_bond_info_list += list(
            mol.GetSubstructMatches(self.amide_bond_pattern)
        )


        for exclude_rotatable_bond_info in exclude_rotatable_bond_info_list:
            exclude_rotatable_bond_info_reversed = tuple(
                reversed(exclude_rotatable_bond_info)
            )
            if exclude_rotatable_bond_info in default_rotatable_bond_info_list:
                default_rotatable_bond_info_list.remove(exclude_rotatable_bond_info)
            elif (
                exclude_rotatable_bond_info_reversed in default_rotatable_bond_info_list
            ):
                default_rotatable_bond_info_list.remove(
                    exclude_rotatable_bond_info_reversed
                )

        return default_rotatable_bond_info_list
