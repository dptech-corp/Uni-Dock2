import os
import dill as pickle
from rdkit import Chem
import xml.etree.ElementTree as ET

from unidock2.torsion_library import utils


class TorsionLibraryBuilder(object):
    def __init__(self, working_dir_name):
        self.torsion_library_xml_file_name = os.path.join(
            os.path.dirname(__file__), "data", "torsion_library_2020.xml"
        )
        self.working_dir_name = os.path.abspath(working_dir_name)
        self.torsion_library_pickle_file_name = os.path.join(
            self.working_dir_name, "torsion_library.pkl"
        )

        self.torsion_library_tree = ET.parse(self.torsion_library_xml_file_name)

    def build_torsion_library(self):
        torsion_library_dict = {}
        torsion_library_dict["specific_class_name_list"] = []
        torsion_library_dict["specific_class_smarts_list"] = []
        torsion_library_dict["specific_class_pattern_mol_list"] = []
        torsion_library_dict["specific_class_atom_map_dict_list"] = []
        torsion_library_dict["specific_class_node_list"] = []

        hierarchy_class_node_list = self.torsion_library_tree.findall("hierarchyClass")

        for hierarchy_class_node in hierarchy_class_node_list:
            hierarchy_class_name = hierarchy_class_node.get("name")
            hierarchy_class_smarts = hierarchy_class_node.get("smarts")
            hierarchy_class_pattern_mol = Chem.MolFromSmarts(hierarchy_class_smarts)
            hierarchy_class_atom_map_dict = utils.get_pattern_atom_mapping(
                hierarchy_class_pattern_mol
            )

            if hierarchy_class_pattern_mol is None:
                continue

            if hierarchy_class_name == "GG":
                torsion_library_dict["generic_class_name"] = hierarchy_class_name
                torsion_library_dict["generic_class_smarts"] = hierarchy_class_smarts
                torsion_library_dict["generic_class_pattern_mol"] = (
                    hierarchy_class_pattern_mol
                )
                torsion_library_dict["generic_class_atom_map_dict"] = (
                    hierarchy_class_atom_map_dict
                )
                torsion_library_dict["generic_class_node"] = hierarchy_class_node

            else:
                torsion_library_dict["specific_class_name_list"].append(
                    hierarchy_class_name
                )
                torsion_library_dict["specific_class_smarts_list"].append(
                    hierarchy_class_smarts
                )
                torsion_library_dict["specific_class_pattern_mol_list"].append(
                    hierarchy_class_pattern_mol
                )
                torsion_library_dict["specific_class_atom_map_dict_list"].append(
                    hierarchy_class_atom_map_dict
                )
                torsion_library_dict["specific_class_node_list"].append(
                    hierarchy_class_node
                )

        ########################################################################################
        ## Iterate analysis for all generic and specific hierarchy torsion class
        num_specific_class = len(torsion_library_dict["specific_class_node_list"])
        torsion_library_dict["specific_class_node_info_list"] = [
            None
        ] * num_specific_class

        for class_idx in range(num_specific_class):
            self.node_idx = 0
            torsion_class_node = torsion_library_dict["specific_class_node_list"][
                class_idx
            ]
            torsion_class_info_dict = {}
            torsion_class_info_dict["node_idx"] = self.node_idx
            torsion_class_info_dict["type"] = "class"
            torsion_class_info_dict["name"] = torsion_class_node.get("name")
            torsion_class_info_dict["smarts"] = torsion_class_node.get("smarts")
            torsion_class_info_dict["pattern_mol"] = Chem.MolFromSmarts(
                torsion_class_node.get("smarts")
            )
            torsion_class_info_dict["atom_map_dict"] = utils.get_pattern_atom_mapping(
                torsion_class_info_dict["pattern_mol"]
            )
            torsion_class_info_dict["node_info_list"] = []

            for node in torsion_class_node:
                self.__collect_torsion_rules__(node, torsion_class_info_dict)

            torsion_library_dict["specific_class_node_info_list"][class_idx] = (
                torsion_class_info_dict
            )

        self.node_idx = 0
        torsion_class_node = torsion_library_dict["generic_class_node"]
        torsion_class_info_dict = {}
        torsion_class_info_dict["node_idx"] = self.node_idx
        torsion_class_info_dict["type"] = "class"
        torsion_class_info_dict["name"] = torsion_class_node.get("name")
        torsion_class_info_dict["smarts"] = torsion_class_node.get("smarts")
        torsion_class_info_dict["pattern_mol"] = Chem.MolFromSmarts(
            torsion_class_node.get("smarts")
        )
        torsion_class_info_dict["atom_map_dict"] = utils.get_pattern_atom_mapping(
            torsion_class_info_dict["pattern_mol"]
        )
        torsion_class_info_dict["node_info_list"] = []

        for node in torsion_class_node:
            self.__collect_torsion_rules__(node, torsion_class_info_dict)

        torsion_library_dict["generic_class_node_info"] = torsion_class_info_dict
        ########################################################################################

        self.torsion_library_dict = torsion_library_dict
        with open(self.torsion_library_pickle_file_name, "wb") as f:
            pickle.dump(torsion_library_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __collect_torsion_rules__(self, node, parent_node_info_dict):
        self.node_idx += 1
        node.set("node_idx", self.node_idx)

        if node.tag == "hierarchySubClass":
            node_info_dict = {}
            node_info_dict["node_idx"] = node.get("node_idx")
            node_info_dict["type"] = "subclass"
            node_info_dict["name"] = node.get("name")
            node_info_dict["smarts"] = node.get("smarts")
            node_info_dict["pattern_mol"] = Chem.MolFromSmarts(node.get("smarts"))
            node_info_dict["atom_map_dict"] = utils.get_pattern_atom_mapping(
                node_info_dict["pattern_mol"]
            )
            node_info_dict["node_info_list"] = []

            num_offspring_nodes = len(node)
            for offspring_node_idx in range(num_offspring_nodes):
                offspring_node = node[offspring_node_idx]
                self.node_idx += 1
                offspring_node.set("node_idx", self.node_idx)
                self.__collect_torsion_rules__(offspring_node, node_info_dict)

            parent_node_info_dict["node_info_list"].append(node_info_dict)

        elif node.tag == "torsionRule":
            node_info_dict = utils.analyze_torsion_rule(node)
            if node_info_dict is not None:
                parent_node_info_dict["node_info_list"].append(node_info_dict)
