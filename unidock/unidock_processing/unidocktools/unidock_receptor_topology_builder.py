import os
from shutil import which
import msys

from unidock_processing.utils.molecule_processing import get_mol_without_indices
from unidock_processing.unidocktools.protein_topology import (
    prepare_receptor_residue_mol_list,
)
from unidock_processing.unidocktools.receptor_topology_preparation import (
    ReceptorTopologyPreparation,
)
from unidock_processing.unidocktools.unidock_vina_atom_types import VINA_ATOM_TYPE_DICT
from unidock_processing.unidocktools.unidock_ff_atom_types import FF_ATOM_TYPE_DICT

class UnidockReceptorTopologyBuilder(object):
    def __init__(
        self,
        receptor_file_name,
        prepared_hydrogen=False,
        covalent_residue_atom_info_list=None,
        working_dir_name=".",
    ):
        self.receptor_file_name = receptor_file_name
        self.prepared_hydrogen = prepared_hydrogen
        self.covalent_residue_atom_info_list = covalent_residue_atom_info_list

        self.working_dir_name = os.path.abspath(working_dir_name)
        self.receptor_structure_dms_file_name = os.path.join(
            self.working_dir_name, "receptor_structure.dms"
        )
        self.receptor_parameterized_dms_file_name = os.path.join(
            self.working_dir_name, "receptor_parameterized.dms"
        )

    def run_protein_preparation(self):
        if which("fepfixer") is not None and which("utop") is not None:
            if self.prepared_hydrogen:
                fep_fixer_command = (
                    f"fepfixer -i {self.receptor_file_name} "
                    f"-o {self.receptor_structure_dms_file_name} "
                    "--custom-protonation-states"
                )
            else:
                fep_fixer_command = (
                    f"fepfixer -i {self.receptor_file_name} "
                    f"-o {self.receptor_structure_dms_file_name}"
                )

            unitop_command = (
                f"utop prm -i {self.receptor_structure_dms_file_name} "
                f"-o {self.receptor_parameterized_dms_file_name}"
            )

            os.system(fep_fixer_command)
            os.system(unitop_command)
        else:
            receptor_topology_preparation = ReceptorTopologyPreparation(
                self.receptor_file_name, self.working_dir_name
            )
            receptor_topology_preparation.run_preparation()

    def find_covalent_hydrogen_atoms(self, atom):
        for neighbor_atom in atom.GetNeighbors():
            neighbor_atom_idx = neighbor_atom.GetIdx()
            neighbor_atom_name = neighbor_atom.GetProp("atom_name")

            if neighbor_atom_name.startswith("H"):
                self.covalent_residue_atom_idx_list.append(neighbor_atom_idx)

    def prepare_covalent_bond_on_residue(self):
        covalent_anchor_atom_info = tuple(self.covalent_residue_atom_info_list[0])
        covalent_bond_start_atom_info = tuple(self.covalent_residue_atom_info_list[1])
        covalent_bond_end_atom_info = tuple(self.covalent_residue_atom_info_list[2])

        num_protein_residues = len(self.protein_residue_property_mol_list)
        for residue_idx in range(num_protein_residues):
            residue_mol = self.protein_residue_property_mol_list[residue_idx]
            atom = residue_mol.GetAtomWithIdx(0)
            chain_idx = atom.GetProp("chain_idx")
            resname = atom.GetProp("residue_name")
            resid = atom.GetIntProp("residue_idx")
            atom_info = (chain_idx, resname, resid)

            if atom_info == covalent_anchor_atom_info[:3]:
                self.covalent_residue_idx = residue_idx
                break

        if self.covalent_residue_idx is None:
            raise ValueError("Cannot find covalent residues from user inputs!!")

        covalent_residue_mol = self.protein_residue_property_mol_list[
            self.covalent_residue_idx
        ]
        num_residue_atoms = covalent_residue_mol.GetNumAtoms()

        for atom_idx in range(num_residue_atoms):
            atom = covalent_residue_mol.GetAtomWithIdx(atom_idx)
            chain_idx = atom.GetProp("chain_idx")
            resname = atom.GetProp("residue_name")
            resid = atom.GetIntProp("residue_idx")
            atom_name = atom.GetProp("atom_name")
            atom_info = (chain_idx, resname, resid, atom_name)

            if atom_info == covalent_anchor_atom_info:
                self.covalent_residue_atom_idx_list.append(atom_idx)
                self.find_covalent_hydrogen_atoms(atom)

            elif atom_info == covalent_bond_start_atom_info:
                self.covalent_residue_atom_idx_list.append(atom_idx)
                self.find_covalent_hydrogen_atoms(atom)

            elif atom_info == covalent_bond_end_atom_info:
                self.covalent_residue_atom_idx_list.append(atom_idx)
                self.find_covalent_hydrogen_atoms(atom)

        processed_covalent_residue_mol = get_mol_without_indices(
            covalent_residue_mol,
            remove_indices=self.covalent_residue_atom_idx_list,
            keep_properties=[
                "atom_idx",
                "atom_name",
                "atom_charge",
                "ff_atom_type",
                "vina_atom_type",
                "residue_idx",
                "residue_name",
                "chain_idx",
                "internal_atom_idx",
                "internal_residue_idx",
                "x",
                "y",
                "z",
            ],
        )

        self.protein_residue_property_mol_list[self.covalent_residue_idx] = (
            processed_covalent_residue_mol
        )

    def generate_receptor_topology(self):
        receptor_file_extension = self.receptor_file_name.split(".")[-1]
        if receptor_file_extension == "pdb":
            self.run_protein_preparation()
        elif receptor_file_extension == "dms":
            self.receptor_parameterized_dms_file_name = self.receptor_file_name
        else:
            raise ValueError(
                "Only PDB and DMS are supported for receptor file extensions!!"
            )

    def analyze_receptor_topology(self):
        receptor_msys_system = msys.LoadDMS(self.receptor_parameterized_dms_file_name)
        (
            self.protein_property_mol,
            self.protein_residue_property_mol_list,
            self.cofactor_residue_property_mol_list,
        ) = prepare_receptor_residue_mol_list(receptor_msys_system)

        self.covalent_residue_idx = None
        self.covalent_residue_atom_idx_list = []
        if self.covalent_residue_atom_info_list is not None:
            self.prepare_covalent_bond_on_residue()

        num_protein_residues = len(self.protein_residue_property_mol_list)
        num_cofactor_residues = len(self.cofactor_residue_property_mol_list)

        num_protein_atoms = 0
        for protein_residue_idx in range(num_protein_residues):
            protein_residue_property_mol = self.protein_residue_property_mol_list[
                protein_residue_idx
            ]
            num_protein_atoms += protein_residue_property_mol.GetNumAtoms()

        num_cofactor_atoms = 0
        for cofactor_residue_idx in range(num_cofactor_residues):
            cofactor_residue_property_mol = self.cofactor_residue_property_mol_list[
                cofactor_residue_idx
            ]
            num_cofactor_atoms += cofactor_residue_property_mol.GetNumAtoms()

        num_receptor_atoms = num_protein_atoms + num_cofactor_atoms
        self.atom_info_nested_list = [None] * num_receptor_atoms
        atom_idx = 0

        for protein_residue_idx in range(num_protein_residues):
            protein_residue_property_mol = self.protein_residue_property_mol_list[
                protein_residue_idx
            ]
            for atom in protein_residue_property_mol.GetAtoms():
                ff_atom_type = atom.GetProp("ff_atom_type")
                vina_atom_type = atom.GetProp("vina_atom_type")

                atom_info_list = [None] * 6
                atom_info_list[0] = atom.GetDoubleProp("x")
                atom_info_list[1] = atom.GetDoubleProp("y")
                atom_info_list[2] = atom.GetDoubleProp("z")
                atom_info_list[3] = VINA_ATOM_TYPE_DICT[vina_atom_type]
                atom_info_list[4] = FF_ATOM_TYPE_DICT[ff_atom_type]
                atom_info_list[5] = atom.GetDoubleProp("atom_charge")

                self.atom_info_nested_list[atom_idx] = atom_info_list
                atom_idx += 1

        for cofactor_residue_idx in range(num_cofactor_residues):
            cofactor_residue_property_mol = self.cofactor_residue_property_mol_list[
                cofactor_residue_idx
            ]
            for atom in cofactor_residue_property_mol.GetAtoms():
                ff_atom_type = atom.GetProp("ff_atom_type")
                vina_atom_type = atom.GetProp("vina_atom_type")

                atom_info_list = [None] * 6
                atom_info_list[0] = atom.GetDoubleProp("x")
                atom_info_list[1] = atom.GetDoubleProp("y")
                atom_info_list[2] = atom.GetDoubleProp("z")
                atom_info_list[3] = VINA_ATOM_TYPE_DICT[vina_atom_type]
                atom_info_list[4] = FF_ATOM_TYPE_DICT[ff_atom_type]
                atom_info_list[5] = atom.GetDoubleProp("atom_charge")

                self.atom_info_nested_list[atom_idx] = atom_info_list
                atom_idx += 1

    def get_summary_receptor_info_dict(self):
        self.receptor_info_summary_dict = {}
        self.receptor_info_summary_dict["receptor"] = self.atom_info_nested_list
