from copy import deepcopy

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D

from unidock.unidock_processing.torsion_library.utils import get_torsion_atom_idx_tuple, get_torsion_mobile_atom_idx_list, rotate_torsion_angle
from unidock.unidock_processing.torsion_library.torsion_rule_matcher import TorsionRuleMatcher

class TorsionLibraryDriver(object):
    def __init__(self,
                 mol,
                 rotatable_bond_info_list,
                 torsion_library_dict):

        self.mol = deepcopy(mol)
        self.rotatable_bond_info_list = rotatable_bond_info_list
        self.num_rotatable_bonds = len(self.rotatable_bond_info_list)
        self.torsion_library_dict = torsion_library_dict

    def perform_torsion_matches(self):
        torsion_rule_matcher = TorsionRuleMatcher(self.mol,
                                                  self.rotatable_bond_info_list,
                                                  self.torsion_library_dict)

        torsion_rule_matcher.match_torsion_rules()
        self.matched_torsion_info_dict_list = torsion_rule_matcher.matched_torsion_info_dict_list

        #############################################################################################
        ## Enumerate all torsion values
        self.torsion_atom_idx_nested_list = [None] * self.num_rotatable_bonds
        self.enumerated_torsion_value_nested_list = [None] * self.num_rotatable_bonds
        self.enumerated_torsion_range_nested_list = [None] * self.num_rotatable_bonds
        self.original_torsion_value_list = [None] * self.num_rotatable_bonds

        conformer = self.mol.GetConformer()

        for rotatable_bond_idx in range(self.num_rotatable_bonds):
            rotatable_bond_info = self.rotatable_bond_info_list[rotatable_bond_idx]

            for matched_torsion_info_dict in self.matched_torsion_info_dict_list:
                current_rotatable_bond_info = matched_torsion_info_dict['rotatable_bond_info']
                if current_rotatable_bond_info == rotatable_bond_info:
                    torsion_atom_idx_list = matched_torsion_info_dict['torsion_atom_idx']
                    self.torsion_atom_idx_nested_list[rotatable_bond_idx] = torsion_atom_idx_list
                    self.original_torsion_value_list[rotatable_bond_idx] = Chem.rdMolTransforms.GetDihedralDeg(conformer, *torsion_atom_idx_list)
                    self.enumerated_torsion_value_nested_list[rotatable_bond_idx] = matched_torsion_info_dict['torsion_angle_value']
                    self.enumerated_torsion_range_nested_list[rotatable_bond_idx] = matched_torsion_info_dict['torsion_angle_range']

            if not self.torsion_atom_idx_nested_list[rotatable_bond_idx]:
                torsion_atom_idx_list = get_torsion_atom_idx_tuple(self.mol, rotatable_bond_info)
                self.torsion_atom_idx_nested_list[rotatable_bond_idx] = torsion_atom_idx_list
                self.original_torsion_value_list[rotatable_bond_idx] = Chem.rdMolTransforms.GetDihedralDeg(conformer, *torsion_atom_idx_list)
                self.enumerated_torsion_value_nested_list[rotatable_bond_idx] = [-150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150, 180.0]
                self.enumerated_torsion_range_nested_list[rotatable_bond_idx] = [(-150.0, 180.0)]
        #############################################################################################

    def identify_torsion_mobile_atoms(self, root_atom_idx):
        self.mobile_atom_idx_nested_list = [None] * self.num_rotatable_bonds
        for rotatable_bond_idx in range(self.num_rotatable_bonds):
            rotatable_bond_info = self.rotatable_bond_info_list[rotatable_bond_idx]
            mobile_atom_idx_list = get_torsion_mobile_atom_idx_list(self.mol, rotatable_bond_info, root_atom_idx)
            self.mobile_atom_idx_nested_list[rotatable_bond_idx] = mobile_atom_idx_list

            #########################################################################################
            #########################################################################################
            ## reorder atom idx list so that it starts from root to the rotamer
            torsion_atom_idx_list = self.torsion_atom_idx_nested_list[rotatable_bond_idx]
            torsion_atom_idx_i = torsion_atom_idx_list[0]
            if torsion_atom_idx_i in mobile_atom_idx_list:
                reordered_torsion_atom_idx_list = list(reversed(torsion_atom_idx_list))
            else:
                reordered_torsion_atom_idx_list = torsion_atom_idx_list

            self.torsion_atom_idx_nested_list[rotatable_bond_idx] = reordered_torsion_atom_idx_list
            #########################################################################################
            #########################################################################################
