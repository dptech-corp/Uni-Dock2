import os
import dill as pickle
import numpy as np
from MDAnalysis.lib.transformations import rotation_matrix as get_rotation_matrix
from MDAnalysis.lib.distances import calc_dihedrals
from rdkit import Chem
from rdkit.Chem import GetMolFrags, FragmentOnBonds
from rdkit.Geometry.rdGeometry import Point3D
from rdkit import RDLogger


rdlg = RDLogger.logger()
rdlg.setLevel(RDLogger.CRITICAL)


def get_pattern_atom_mapping(pattern_mol):
    atom_map_dict = {}
    num_pattern_atoms = pattern_mol.GetNumAtoms()
    for atom_idx in range(num_pattern_atoms):
        atom = pattern_mol.GetAtomWithIdx(atom_idx)
        if atom.HasProp("molAtomMapNumber"):
            atom_map_number = atom.GetIntProp("molAtomMapNumber")
            atom_map_dict[atom_map_number] = atom_idx

    return atom_map_dict


def canonicalize_angle(angle):
    if angle > 180.0:
        canonical_angle = angle - 360.0
    elif angle <= -180.0:
        canonical_angle = angle + 360.0
    else:
        canonical_angle = angle

    return canonical_angle


def get_detailed_torsion_ranges(torsion_range_list):
    detailed_torsion_range_list = []
    for torsion_range_tuple in torsion_range_list:
        torsion_lower_bound = torsion_range_tuple[0]
        torsion_upper_bound = torsion_range_tuple[1]
        if torsion_lower_bound > torsion_upper_bound:
            detailed_torsion_tuple_list = [
                (torsion_lower_bound, 180.0),
                (-180.0, torsion_upper_bound),
            ]
            detailed_torsion_range_list.extend(detailed_torsion_tuple_list)
        else:
            detailed_torsion_range_list.append(torsion_range_tuple)

    return detailed_torsion_range_list


def get_merged_torsion_ranges(torsion_range_list):
    merged_torsion_range_list = []
    torsion_range_list.sort(key=lambda torsion_range: torsion_range[0])

    for torsion_range in torsion_range_list:
        torsion_lower_bound = torsion_range[0]
        torsion_upper_bound = torsion_range[1]

        if len(merged_torsion_range_list) == 0:
            merged_torsion_range_list.append(torsion_range)
        else:
            checked_torsion_range = merged_torsion_range_list[-1]
            checked_torsion_lower_bound = checked_torsion_range[0]
            checked_torsion_upper_bound = checked_torsion_range[1]

            if checked_torsion_upper_bound < torsion_lower_bound:
                merged_torsion_range_list.append(torsion_range)
            else:
                updated_torsion_range = (
                    checked_torsion_lower_bound,
                    max(checked_torsion_upper_bound, torsion_upper_bound),
                )
                merged_torsion_range_list[-1] = updated_torsion_range

    return merged_torsion_range_list


def analyze_torsion_rule(torsion_rule_node):
    torsion_rule_info_dict = {}
    torsion_smarts = torsion_rule_node.get("smarts")
    torsion_pattern_mol = Chem.MolFromSmarts(torsion_smarts)

    if "N_lp" in torsion_smarts:
        return None

    if torsion_pattern_mol is None:
        return None

    torsion_atom_map_dict = get_pattern_atom_mapping(torsion_pattern_mol)

    torsion_angle_list_node_list = torsion_rule_node.findall("angleList")
    for torsion_angle_list_node in torsion_angle_list_node_list:
        torsion_angle_node_list = torsion_angle_list_node.findall("angle")
        num_torsion_bins = len(torsion_angle_node_list)
        torsion_value_list = [None] * num_torsion_bins
        torsion_tolerance_list = [None] * num_torsion_bins

        for bin_idx in range(num_torsion_bins):
            torsion_info = torsion_angle_node_list[bin_idx].attrib
            torsion_value_list[bin_idx] = float(torsion_info["value"])
            torsion_tolerance_list[bin_idx] = float(torsion_info["tolerance1"])

        torsion_value_set = set(torsion_value_list)
        if set(
            {
                -150.0,
                -120.0,
                -90.0,
                -60.0,
                -30.0,
                0.0,
                30.0,
                60.0,
                90.0,
                120.0,
                150.0,
                180.0,
            }
        ).issubset(torsion_value_set):
            torsion_distribution = "evenly_distributed"
        else:
            torsion_distribution = [None] * num_torsion_bins
            for bin_idx in range(num_torsion_bins):
                torsion_value = canonicalize_angle(torsion_value_list[bin_idx])
                torsion_tolerance = canonicalize_angle(torsion_tolerance_list[bin_idx])
                torsion_lower_bound = canonicalize_angle(
                    torsion_value - torsion_tolerance
                )
                torsion_upper_bound = canonicalize_angle(
                    torsion_value + torsion_tolerance
                )

                torsion_distribution[bin_idx] = (
                    torsion_lower_bound,
                    torsion_upper_bound,
                )

            torsion_distribution = get_detailed_torsion_ranges(torsion_distribution)
            torsion_distribution = get_merged_torsion_ranges(torsion_distribution)

    torsion_histogram_node_list = torsion_rule_node.findall("histogram")
    for torsion_histogram_node in torsion_histogram_node_list:
        torsion_bin_node_list = torsion_histogram_node.findall("bin")
        num_weight_bins = len(torsion_bin_node_list)
        torsion_histogram_weight_list = [None] * num_weight_bins

        for bin_idx in range(num_weight_bins):
            histogram_weight_info = torsion_bin_node_list[bin_idx].attrib
            torsion_histogram_weight_list[bin_idx] = float(
                histogram_weight_info["count"]
            )

    torsion_rule_info_dict["node_idx"] = torsion_rule_node.get("node_idx")
    torsion_rule_info_dict["type"] = "torsion_rule"
    torsion_rule_info_dict["smarts"] = torsion_smarts
    torsion_rule_info_dict["pattern_mol"] = torsion_pattern_mol
    torsion_rule_info_dict["atom_map_dict"] = torsion_atom_map_dict
    torsion_rule_info_dict["angle_value"] = torsion_value_list
    torsion_rule_info_dict["angle_range"] = torsion_distribution
    torsion_rule_info_dict["histogram_weight"] = torsion_histogram_weight_list

    return torsion_rule_info_dict


def get_torsion_atom_idx_tuple(mol, rotatable_bond_info):
    atom_idx_1 = rotatable_bond_info[0]
    atom_idx_2 = rotatable_bond_info[1]

    atom_1 = mol.GetAtomWithIdx(atom_idx_1)
    atom_2 = mol.GetAtomWithIdx(atom_idx_2)

    for atom in atom_1.GetNeighbors():
        atom_idx = atom.GetIdx()
        if atom_idx != atom_idx_2:
            atom_idx_0 = atom_idx
            break

    for atom in atom_2.GetNeighbors():
        atom_idx = atom.GetIdx()
        if atom_idx != atom_idx_1:
            atom_idx_3 = atom_idx
            break

    torsion_atom_idx_tuple = (atom_idx_0, atom_idx_1, atom_idx_2, atom_idx_3)

    return torsion_atom_idx_tuple


def get_torsion_mobile_atom_idx_list(mol, rotatable_bond_info, root_atom_idx):
    bond_list = list(mol.GetBonds())
    for bond in bond_list:
        bond_info = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        bond_info_reversed = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
        if (
            bond_info == rotatable_bond_info
            or bond_info_reversed == rotatable_bond_info
        ):
            bond_idx = bond.GetIdx()
            break

    splitted_mol = FragmentOnBonds(mol, [bond_idx], addDummies=False)
    splitted_mol_list = list(
        GetMolFrags(splitted_mol, asMols=True, sanitizeFrags=False)
    )

    static_fragment = None
    mobile_fragment = None
    for atom in splitted_mol_list[0].GetAtoms():
        atom_idx = atom.GetIntProp("internal_atom_idx")
        if atom_idx == root_atom_idx:
            static_fragment = splitted_mol_list[0]
            mobile_fragment = splitted_mol_list[1]
            break

    if static_fragment is None:
        static_fragment = splitted_mol_list[1]
        mobile_fragment = splitted_mol_list[0]

    num_mobile_atoms = mobile_fragment.GetNumAtoms()
    mobile_atom_idx_list = [None] * num_mobile_atoms

    for idx in range(num_mobile_atoms):
        atom = mobile_fragment.GetAtomWithIdx(idx)
        mobile_atom_idx_list[idx] = atom.GetIntProp("internal_atom_idx")

    return mobile_atom_idx_list


def rotate_torsion_angle(
    mol, torsion_atom_idx_list, mobile_atom_idx_list, torsion_angle
):
    torsion_atom_idx_0 = torsion_atom_idx_list[0]
    torsion_atom_idx_1 = torsion_atom_idx_list[1]
    torsion_atom_idx_2 = torsion_atom_idx_list[2]
    torsion_atom_idx_3 = torsion_atom_idx_list[3]

    target_rotatable_bond_info = (torsion_atom_idx_1, torsion_atom_idx_2)

    conformer = mol.GetConformer()
    positions = conformer.GetPositions()

    torsion_atom_position_0 = positions[torsion_atom_idx_0, :]
    torsion_atom_position_1 = positions[torsion_atom_idx_1, :]
    torsion_atom_position_2 = positions[torsion_atom_idx_2, :]
    torsion_atom_position_3 = positions[torsion_atom_idx_3, :]

    target_torsion_value = np.degrees(
        calc_dihedrals(
            torsion_atom_position_0,
            torsion_atom_position_1,
            torsion_atom_position_2,
            torsion_atom_position_3,
        )
    )

    mobile_positions = positions[mobile_atom_idx_list, :]

    if target_rotatable_bond_info[0] not in mobile_atom_idx_list:
        torsion_bond_position_0 = positions[target_rotatable_bond_info[0], :]
        torsion_bond_position_1 = positions[target_rotatable_bond_info[1], :]
    else:
        torsion_bond_position_0 = positions[target_rotatable_bond_info[1], :]
        torsion_bond_position_1 = positions[target_rotatable_bond_info[0], :]

    dihedral_rotate_axis = torsion_bond_position_1 - torsion_bond_position_0
    unit_dihedral_rotate_axis = dihedral_rotate_axis / np.linalg.norm(
        dihedral_rotate_axis
    )

    delta_torsion_angle = torsion_angle - target_torsion_value
    delta_torsion_angle = np.radians(delta_torsion_angle)

    transformation_matrix = get_rotation_matrix(
        delta_torsion_angle, unit_dihedral_rotate_axis, torsion_bond_position_0
    )
    rotation = transformation_matrix[:3, :3].T
    translation = transformation_matrix[:3, 3]
    transformed_mobile_positions = np.dot(mobile_positions, rotation)
    transformed_mobile_positions += translation

    positions[mobile_atom_idx_list, :] = transformed_mobile_positions

    num_total_atoms = mol.GetNumAtoms()
    for atom_idx in range(num_total_atoms):
        atom_positions = positions[atom_idx, :]
        atom_coord_point_3D = Point3D(
            atom_positions[0], atom_positions[1], atom_positions[2]
        )
        conformer.SetAtomPosition(atom_idx, atom_coord_point_3D)


def get_torsion_lib_dict() -> dict:
    tor_lib_pkl_path = os.path.join(os.path.dirname(__file__), 'data', 'torsion_library.pkl')
    with open(tor_lib_pkl_path, 'rb') as f:
        tor_lib_dict = pickle.load(f)
    return tor_lib_dict
