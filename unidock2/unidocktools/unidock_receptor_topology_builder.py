import os
from itertools import chain
from shutil import which
import msys

from unidock2.utils.external_command import run_external_command
from unidock2.utils.molecule_processing import get_mol_without_indices
from unidock2.unidocktools.protein_topology import (
    prepare_receptor_residue_mol_list,
)
from unidock2.unidocktools.receptor_topology_preparation import (
    ReceptorTopologyPreparation,
)
from unidock2.atom_types.unidock_vina_atom_types import VINA_ATOM_TYPE_DICT
from unidock2.force_field.atom_type_mapping import FF_ATOM_TYPE_DICT


def _receptor_atom_to_engine_record(atom):
    ff_atom_type = atom.GetProp("ff_atom_type")
    vina_atom_type = atom.GetProp("vina_atom_type")
    return [
        atom.GetDoubleProp("x"),
        atom.GetDoubleProp("y"),
        atom.GetDoubleProp("z"),
        VINA_ATOM_TYPE_DICT[vina_atom_type],
        FF_ATOM_TYPE_DICT[ff_atom_type],
        atom.GetDoubleProp("atom_charge"),
    ]


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
            self.working_dir_name, 'receptor_structure.dms'
        )
        self.receptor_parameterized_dms_file_name = os.path.join(
            self.working_dir_name, 'receptor_parameterized.dms'
        )
        self.summary_receptor_info_json_file_name = os.path.join(
            self.working_dir_name, 'summary_receptor_info.json'
        )

    def run_protein_preparation(self):
        fepfixer_executable = which("fepfixer")
        utop_executable = which("utop")
        if fepfixer_executable is not None and utop_executable is not None:
            fepfixer_command = [
                fepfixer_executable,
                "-i",
                os.path.abspath(self.receptor_file_name),
                "-o",
                os.path.basename(self.receptor_structure_dms_file_name),
            ]
            if self.prepared_hydrogen:
                fepfixer_command.append("--custom-protonation-states")

            run_external_command(
                fepfixer_command,
                cwd=self.working_dir_name,
                log_file_name="fepfixer.log",
                expected_output_file_names=[self.receptor_structure_dms_file_name],
            )
            run_external_command(
                [
                    utop_executable,
                    "prm",
                    "-i",
                    os.path.basename(self.receptor_structure_dms_file_name),
                    "-o",
                    os.path.basename(self.receptor_parameterized_dms_file_name),
                ],
                cwd=self.working_dir_name,
                log_file_name="utop.log",
                expected_output_file_names=[self.receptor_parameterized_dms_file_name],
            )
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

        residue_property_mols = chain(
            self.protein_residue_property_mol_list,
            self.cofactor_residue_property_mol_list,
        )
        self.atom_info_nested_list = [
            _receptor_atom_to_engine_record(atom)
            for residue_property_mol in residue_property_mols
            for atom in residue_property_mol.GetAtoms()
        ]

    def get_summary_receptor_info(self):
        self.summary_receptor_info_dict = {}
        self.summary_receptor_info_dict['receptor'] = self.atom_info_nested_list
