class TorsionRuleMatcher(object):
    def __init__(self, mol, rotatable_bond_info_list, torsion_library_dict):
        self.mol = mol
        self.rotatable_bond_info_list = rotatable_bond_info_list
        self.torsion_library_dict = torsion_library_dict

        self.matched_rotatable_bond_info_set = set()
        self.matched_torsion_info_dict_list = []

    def match_torsion_rules(self):
        match_status = False

        ########################################################################################
        ## First match all specific classes
        num_specific_class = len(self.torsion_library_dict["specific_class_name_list"])
        for class_idx in range(num_specific_class):
            class_pattern_mol = self.torsion_library_dict[
                "specific_class_pattern_mol_list"
            ][class_idx]
            class_atom_map_dict = self.torsion_library_dict[
                "specific_class_atom_map_dict_list"
            ][class_idx]

            matched_pattern_idx_tuple_list = list(
                self.mol.GetSubstructMatches(class_pattern_mol)
            )
            if len(matched_pattern_idx_tuple_list) == 0:
                continue

            matched_rotatable_bond_info_list = []
            for matched_pattern_idx_tuple in matched_pattern_idx_tuple_list:
                torsion_atom_begin_idx = matched_pattern_idx_tuple[
                    class_atom_map_dict[2]
                ]
                torsion_atom_end_idx = matched_pattern_idx_tuple[class_atom_map_dict[3]]
                bond_info = (torsion_atom_begin_idx, torsion_atom_end_idx)
                bond_info_reversed = (torsion_atom_end_idx, torsion_atom_begin_idx)

                if bond_info in self.rotatable_bond_info_list:
                    matched_rotatable_bond_info_list.append(bond_info)
                elif bond_info_reversed in self.rotatable_bond_info_list:
                    matched_rotatable_bond_info_list.append(bond_info_reversed)

            if len(matched_rotatable_bond_info_list) == 0:
                continue
            else:
                for node_info_dict in self.torsion_library_dict[
                    "specific_class_node_info_list"
                ][class_idx]["node_info_list"]:
                    match_status = self.__match_torsion_library__(node_info_dict)
                    if match_status is True:
                        break

            if match_status is True:
                break

        # If matching not completed in specific class matches, proceed to generic class match # noqa
        if not match_status:
            class_pattern_mol = self.torsion_library_dict["generic_class_pattern_mol"]
            class_atom_map_dict = self.torsion_library_dict[
                "generic_class_atom_map_dict"
            ]

            matched_pattern_idx_tuple_list = list(
                self.mol.GetSubstructMatches(class_pattern_mol)
            )

            if len(matched_pattern_idx_tuple_list) > 0:
                matched_rotatable_bond_info_list = []

                for matched_pattern_idx_tuple in matched_pattern_idx_tuple_list:
                    torsion_atom_begin_idx = matched_pattern_idx_tuple[
                        class_atom_map_dict[2]
                    ]
                    torsion_atom_end_idx = matched_pattern_idx_tuple[
                        class_atom_map_dict[3]
                    ]
                    bond_info = (torsion_atom_begin_idx, torsion_atom_end_idx)
                    bond_info_reversed = (torsion_atom_end_idx, torsion_atom_begin_idx)

                    if bond_info in self.rotatable_bond_info_list:
                        matched_rotatable_bond_info_list.append(bond_info)
                    elif bond_info_reversed in self.rotatable_bond_info_list:
                        matched_rotatable_bond_info_list.append(bond_info_reversed)

                if len(matched_rotatable_bond_info_list) > 0:
                    for node_info_dict in self.torsion_library_dict[
                        "generic_class_node_info"
                    ]["node_info_list"]:
                        match_status = self.__match_torsion_library__(node_info_dict)
                        if match_status is True:
                            break

    def __match_torsion_library__(self, node_info_dict):
        node_type = node_info_dict["type"]
        node_smarts = node_info_dict["smarts"]
        node_pattern_mol = node_info_dict["pattern_mol"]
        node_atom_map_dict = node_info_dict["atom_map_dict"]

        if node_type == "subclass":
            matched_pattern_idx_tuple_list = list(
                self.mol.GetSubstructMatches(node_pattern_mol)
            )

            if len(matched_pattern_idx_tuple_list) == 0:
                return False

            matched_rotatable_bond_info_list = []
            for matched_pattern_idx_tuple in matched_pattern_idx_tuple_list:
                torsion_atom_begin_idx = matched_pattern_idx_tuple[
                    node_atom_map_dict[2]
                ]
                torsion_atom_end_idx = matched_pattern_idx_tuple[node_atom_map_dict[3]]
                bond_info = (torsion_atom_begin_idx, torsion_atom_end_idx)
                bond_info_reversed = (torsion_atom_end_idx, torsion_atom_begin_idx)

                if bond_info in self.rotatable_bond_info_list:
                    matched_rotatable_bond_info_list.append(bond_info)
                elif bond_info_reversed in self.rotatable_bond_info_list:
                    matched_rotatable_bond_info_list.append(bond_info_reversed)

            if len(matched_rotatable_bond_info_list) == 0:
                return False
            else:
                for offspring_node_info_dict in node_info_dict["node_info_list"]:
                    match_status = self.__match_torsion_library__(
                        offspring_node_info_dict
                    )
                    if match_status is True:
                        return True
                    else:
                        continue

                return False

        elif node_type == "torsion_rule":
            matched_pattern_idx_tuple_list = list(
                self.mol.GetSubstructMatches(node_pattern_mol)
            )

            if len(matched_pattern_idx_tuple_list) == 0:
                return False

            for matched_pattern_idx_tuple in matched_pattern_idx_tuple_list:
                torsion_atom_0_idx = matched_pattern_idx_tuple[node_atom_map_dict[1]]
                torsion_atom_1_idx = matched_pattern_idx_tuple[node_atom_map_dict[2]]
                torsion_atom_2_idx = matched_pattern_idx_tuple[node_atom_map_dict[3]]
                torsion_atom_3_idx = matched_pattern_idx_tuple[node_atom_map_dict[4]]

                bond_info = (torsion_atom_1_idx, torsion_atom_2_idx)
                bond_info_reversed = (torsion_atom_2_idx, torsion_atom_1_idx)
                if bond_info in self.rotatable_bond_info_list:
                    matched_bond_info = bond_info
                elif bond_info_reversed in self.rotatable_bond_info_list:
                    matched_bond_info = bond_info_reversed
                else:
                    continue

                if matched_bond_info in self.matched_rotatable_bond_info_set:
                    continue
                else:
                    self.matched_rotatable_bond_info_set.add(matched_bond_info)

                torsion_atom_idx_tuple = (
                    torsion_atom_0_idx,
                    torsion_atom_1_idx,
                    torsion_atom_2_idx,
                    torsion_atom_3_idx,
                )
                torsion_angle_value = node_info_dict["angle_value"]
                torsion_angle_range = node_info_dict["angle_range"]
                torsion_histogram_weight = node_info_dict["histogram_weight"]

                if torsion_angle_range == "evenly_distributed":
                    continue

                matched_torsion_info_dict = {}
                matched_torsion_info_dict["rotatable_bond_info"] = matched_bond_info
                matched_torsion_info_dict["torsion_atom_idx"] = torsion_atom_idx_tuple
                matched_torsion_info_dict["torsion_smarts"] = node_smarts
                matched_torsion_info_dict["torsion_angle_value"] = torsion_angle_value
                matched_torsion_info_dict["torsion_angle_range"] = torsion_angle_range
                matched_torsion_info_dict["torsion_histogram_weight"] = (
                    torsion_histogram_weight
                )

                self.matched_torsion_info_dict_list.append(matched_torsion_info_dict)

                if len(self.matched_rotatable_bond_info_set) == len(
                    self.rotatable_bond_info_list
                ):
                    return True

            return False
