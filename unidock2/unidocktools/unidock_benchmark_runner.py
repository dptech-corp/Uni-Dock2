import os
import yaml
from shutil import rmtree
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from unidock2.unidocktools.unidock_ligand_pose_writer import (
    UnidockLigandPoseWriter,
)


class UnidockBenchmarkRunner(object):
    def __init__(
        self,
        ligand_crystal_sdf_file_name,
        ligand_sdf_file_name_list,
        unidock2_input_json_file_name,
        target_center,
        option_yaml_file_name=None,
    ):
        self.ligand_crystal_sdf_file_name = ligand_crystal_sdf_file_name
        self.crystal_mol = Chem.SDMolSupplier(
            self.ligand_crystal_sdf_file_name, removeHs=True
        )[0]
        self.atom_mapping_array = np.array(
            self.crystal_mol.GetSubstructMatches(
                self.crystal_mol, useChirality=True, uniquify=False
            )
        )

        crystal_conformer = self.crystal_mol.GetConformer()
        self.crystal_positions = crystal_conformer.GetPositions()

        self.ligand_sdf_file_name_list = ligand_sdf_file_name_list
        self.ligand_mol_list = []

        for ligand_sdf_file_name in self.ligand_sdf_file_name_list:
            ligand_mol_list = list(
                Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False)
            )
            for ligand_mol in ligand_mol_list:
                if ligand_mol is None:
                    logging.error("Incorrect bond orders for molecule!")
                    continue
                if Descriptors.NumRadicalElectrons(ligand_mol) > 0:
                    logging.error("Molecule contains atoms with radicals!")
                    continue

                self.ligand_mol_list.append(ligand_mol)

        self.unidock2_input_json_file_name = unidock2_input_json_file_name
        self.target_center = target_center

        if option_yaml_file_name is not None:
            self.option_yaml_file_name = option_yaml_file_name
        else:
            self.option_yaml_file_name = os.path.join(
                os.path.dirname(__file__), "data", "unidock_option_template.yaml"
            )

        with open(self.option_yaml_file_name, "r") as option_yaml_file:
            self.unidock2_option_dict = yaml.safe_load(option_yaml_file)

        self.template_docking = self.unidock2_option_dict["Preprocessing"][
            "template_docking"
        ]
        self.covalent_ligand = self.unidock2_option_dict["Preprocessing"][
            "covalent_ligand"
        ]

        self.working_dir_name = os.path.abspath(
            self.unidock2_option_dict["Preprocessing"]["working_dir_name"]
        )
        self.unidock2_output_working_dir_name = os.path.join(
            self.working_dir_name, "unidock2_output"
        )

        if os.path.isdir(self.unidock2_output_working_dir_name):
            rmtree(self.unidock2_output_working_dir_name, ignore_errors=True)
            os.mkdir(self.unidock2_output_working_dir_name)
        else:
            os.mkdir(self.unidock2_output_working_dir_name)

    def __prepare_unidock2_input_yaml__(self):
        unidock2_input_dict = {}
        unidock2_input_dict["Advanced"] = self.unidock2_option_dict["Advanced"]
        unidock2_input_dict["Hardware"] = self.unidock2_option_dict["Hardware"]

        unidock2_input_dict["Inputs"] = {}
        unidock2_input_dict["Inputs"]["json"] = self.unidock2_input_json_file_name

        unidock2_input_dict["Outputs"] = {}
        unidock2_input_dict["Outputs"]["dir"] = self.unidock2_output_working_dir_name

        unidock2_input_dict["Settings"] = self.unidock2_option_dict["Settings"]
        unidock2_input_dict["Settings"]["center_x"] = self.target_center[0]
        unidock2_input_dict["Settings"]["center_y"] = self.target_center[1]
        unidock2_input_dict["Settings"]["center_z"] = self.target_center[2]
        unidock2_input_dict["Settings"]["constraint_docking"] = (
            self.template_docking or self.covalent_ligand
        )

        self.unidock2_input_yaml_file_name = os.path.join(
            self.working_dir_name, "system_inputs_unidock2.yaml"
        )
        with open(self.unidock2_input_yaml_file_name, "w") as unidock2_input_yaml_file:
            yaml.dump(unidock2_input_dict, unidock2_input_yaml_file)

    def __calculate_best_matched_rmsd__(
        crystal_coords_array, coords_array, atom_mapping_array
    ):
        num_atoms = coords_array.shape[0]
        num_atom_orders = atom_mapping_array.shape[0]

        min_rmsd = np.inf
        for atom_order_idx in range(num_atom_orders):
            atom_mapping = atom_mapping_array[atom_order_idx, :]
            reordered_coords_array = coords_array[atom_mapping, :]
            rmsd = np.sqrt(
                np.sum((reordered_coords_array - crystal_coords_array) ** 2) / num_atoms
            )
            if rmsd < min_rmsd:
                min_rmsd = rmsd

        return min_rmsd

    def run_unidock_benchmark(self):
        ## write params yaml
        self.__prepare_unidock2_input_yaml__()

        ## run ud2 engine
        os.system(f"ud2 {self.unidock2_input_yaml_file_name}")

        ## generate output ud2 pose sdf
        unidock2_pose_json_file_name_raw_list = os.listdir(
            self.unidock2_output_working_dir_name
        )
        unidock2_pose_json_file_name_list = [
            os.path.join(
                self.unidock2_output_working_dir_name, unidock2_pose_json_file_name_raw
            )
            for unidock2_pose_json_file_name_raw \
                in unidock2_pose_json_file_name_raw_list
        ]
        unidock_pose_writer = UnidockLigandPoseWriter(
            self.ligand_mol_list,
            unidock2_pose_json_file_name_list,
            self.working_dir_name,
        )

        unidock_pose_writer.generate_docking_pose_sdf()

        unidock_pose_mol_list = list(
            Chem.SDMolSupplier(
                unidock_pose_writer.docking_pose_sdf_file_name, removeHs=True
            )
        )
        num_docked_poses = len(unidock_pose_mol_list)
        unidock_pose_rmsd_list = [None] * num_docked_poses

        for pose_idx in range(num_docked_poses):
            unidock_pose_mol = unidock_pose_mol_list[pose_idx]
            unidock_pose_conformer = unidock_pose_mol.GetConformer()
            unidock_pose_positions = unidock_pose_conformer.GetPositions()
            unidock_pose_rmsd = self.__calculate_best_matched_rmsd__(
                self.crystal_positions, unidock_pose_positions, self.atom_mapping_array
            )
            unidock_pose_rmsd_list[pose_idx] = unidock_pose_rmsd

        return unidock_pose_rmsd_list
