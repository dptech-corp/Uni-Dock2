import os
import json
from copy import deepcopy

import numpy as np

from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point3D

from unidock_processing.ligand_topology import utils

class UnidockLigandPoseWriter(object):
    def __init__(
        self,
        ligand_mol_list,
        unidock2_pose_json_file_name_list,
        covalent_ligand=False,
        docking_pose_sdf_file_name='unidock2_pose.sdf',
    ):
        self.ligand_mol_list = ligand_mol_list
        self.num_ligands = len(self.ligand_mol_list)
        self.unidock2_pose_json_file_name_list = unidock2_pose_json_file_name_list
        self.num_unidock2_batches = len(self.unidock2_pose_json_file_name_list)
        self.covalent_ligand = covalent_ligand
        self.docking_pose_sdf_file_name = os.path.abspath(docking_pose_sdf_file_name)

        self.unidock2_pose_dict = {}
        for batch_idx in range(self.num_unidock2_batches):
            unidock2_pose_json_file_name = self.unidock2_pose_json_file_name_list[
                batch_idx
            ]
            with open(unidock2_pose_json_file_name, 'r') as unidockd2_json_file:
                batch_unidock2_pose_dict = json.load(unidockd2_json_file)

            self.unidock2_pose_dict.update(batch_unidock2_pose_dict)

    def generate_docking_pose_sdf(self):
        self.docking_pose_writer = Chem.SDWriter(self.docking_pose_sdf_file_name)

        for ligand_idx in range(self.num_ligands):
            ligand_mol = self.ligand_mol_list[ligand_idx]

            if self.covalent_ligand:
                ligand_mol, covalent_anchor_atom_info, covalent_atom_info_list = (
                    utils.prepare_covalent_ligand_mol(ligand_mol)
                )

            ligand_name = ligand_mol.GetProp('_Name')
            num_ligand_atoms = ligand_mol.GetNumAtoms()
            ligand_unidock2_pose_list = self.unidock2_pose_dict[ligand_name]
            num_poses = len(ligand_unidock2_pose_list)

            for pose_idx in range(num_poses):
                ligand_mol_ud2_pose = deepcopy(ligand_mol)
                ligand_pose_name = f'{ligand_name}_unidock2_pose_{pose_idx}'
                ligand_mol_ud2_pose.SetProp('_Name', ligand_pose_name)

                unidock2_pose_info_dict = ligand_unidock2_pose_list[pose_idx]
                vina_scoring_list = unidock2_pose_info_dict['energy']
                ligand_mol_ud2_pose.SetDoubleProp(
                    'vina_binding_free_energy', float(vina_scoring_list[0])
                )
                ligand_mol_ud2_pose.SetDoubleProp(
                    'vina_intra_inter', float(vina_scoring_list[1])
                )
                ligand_mol_ud2_pose.SetDoubleProp(
                    'vina_intra', float(vina_scoring_list[2])
                )
                ligand_mol_ud2_pose.SetDoubleProp(
                    'vina_inter', float(vina_scoring_list[3])
                )
                ligand_mol_ud2_pose.SetDoubleProp(
                    'vina_box_penalty', float(vina_scoring_list[4])
                )
                ligand_mol_ud2_pose.SetDoubleProp(
                    'vina_torsion_number_energy', float(vina_scoring_list[5])
                )

                ligand_mol_ud2_conf = ligand_mol_ud2_pose.GetConformer()
                ud2_pose_coord_list = unidock2_pose_info_dict['coords']

                total_num_coords = len(ud2_pose_coord_list)
                num_coords_atoms = total_num_coords / 3

                if num_coords_atoms != num_ligand_atoms:
                    raise ValueError(
                        f'Unidock2 output poses do not have equal number of atoms \
                            compared to original molecule on {ligand_pose_name}!!'
                    )

                ud2_pose_coord_array = np.reshape(
                    ud2_pose_coord_list, (num_ligand_atoms, 3)
                )

                for atom_idx in range(num_ligand_atoms):
                    ud2_atom_coord = ud2_pose_coord_array[atom_idx, :]
                    ud2_atom_coord_point_3D = Point3D(
                        float(ud2_atom_coord[0]),
                        float(ud2_atom_coord[1]),
                        float(ud2_atom_coord[2]),
                    )
                    ligand_mol_ud2_conf.SetAtomPosition(
                        atom_idx, ud2_atom_coord_point_3D
                    )

                self.docking_pose_writer.write(ligand_mol_ud2_pose)
                self.docking_pose_writer.flush()

        self.docking_pose_writer.close()
