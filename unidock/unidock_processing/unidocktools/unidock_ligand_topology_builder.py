from typing import Optional
import os
import math
import json
import logging

from multiprocess.pool import Pool
from rdkit import Chem
from rdkit.Chem import Descriptors

from unidock_processing.torsion_library.utils import get_torsion_lib_dict
from unidock_processing.ligand_topology.mol_graph import BaseMolGraph

def batch_topology_builder_process(
    ligand_sdf_file_name: str,
    covalent_ligand: bool,
    template_docking: bool,
    reference_sdf_file_name: str,
    core_atom_mapping_dict_list: list[dict],
    working_dir_name: str,
    construct_ff: bool,
    atom_mapper_align: bool = False,
):
    torsion_library_dict = get_torsion_lib_dict()

    batch_ligand_mol_list = list(Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False))

    mol_graph_type = 'generic'
    if covalent_ligand:
        mol_graph_type = 'covalent'
    if template_docking:
        mol_graph_type = 'template' if not atom_mapper_align else 'atom_mapper_align'
        reference_mol = Chem.SDMolSupplier(reference_sdf_file_name, removeHs=False)[0]
    else:
        reference_mol = None

    num_batch_ligands = len(batch_ligand_mol_list)
    ligand_info_dict_list = [None] * num_batch_ligands
    for ligand_idx in range(num_batch_ligands):
        ligand_mol = batch_ligand_mol_list[ligand_idx]
        ligand_name = ligand_mol.GetProp('ud2_molecule_name')
        core_atom_mapping_dict = core_atom_mapping_dict_list[ligand_idx]
        mol_graph_builder = BaseMolGraph.create(
            mol_graph_type,
            mol=ligand_mol,
            torsion_library_dict=torsion_library_dict,
            reference_mol=reference_mol,
            core_atom_mapping_dict=core_atom_mapping_dict,
            construct_ff=construct_ff,
            working_dir_name=working_dir_name,
        )
        (
            atom_info_nested_list,
            torsion_info_nested_list,
            root_atom_idx_list,
            fragment_atom_idx_nested_list,
        ) = mol_graph_builder.build_graph()

        ligand_info_dict = {}
        ligand_info_dict['ligand_name'] = ligand_name
        ligand_info_dict['atom_info'] = atom_info_nested_list
        ligand_info_dict['torsion_info'] = torsion_info_nested_list
        ligand_info_dict['root_atom_idx'] = root_atom_idx_list
        ligand_info_dict['fragment_atom_idx'] = fragment_atom_idx_nested_list
        ligand_info_dict_list[ligand_idx] = ligand_info_dict

    return ligand_info_dict_list


class UnidockLigandTopologyBuilder(object):
    def __init__(
        self,
        ligand_sdf_file_name_list:list[str],
        covalent_ligand:bool=False,
        template_docking:bool=False,
        reference_sdf_file_name:Optional[str]=None,
        core_atom_mapping_dict_list:Optional[list[dict]]=None,
        n_cpu:Optional[int]=None,
        working_dir_name:str='.',
        construct_ff: bool=False,
        atom_mapper_align:bool=False,
    ):
        self.ligand_sdf_file_name_list = ligand_sdf_file_name_list
        self.ligand_mol_list = []
        valid_mol_num = 0
        for ligand_sdf_file_name in self.ligand_sdf_file_name_list:
            for source_mol_idx, ligand_mol in enumerate(Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False)):
                if ligand_mol is None:
                    logging.error("Incorrect bond orders for molecule!")
                    continue
                if Descriptors.NumRadicalElectrons(ligand_mol) > 0:
                    logging.error("Molecule contains atoms with radicals!")
                    continue

                ligand_mol.SetProp('source_sdf_file_name', ligand_sdf_file_name)
                ligand_mol.SetIntProp('source_mol_idx', source_mol_idx)
                internal_molecule_name = f'MOL_{valid_mol_num}'
                ligand_mol.SetProp('ud2_molecule_name', internal_molecule_name)
                valid_mol_num += 1
                self.ligand_mol_list.append(ligand_mol)

        self.num_ligands = len(self.ligand_mol_list)
        if self.num_ligands == 0:
            raise ValueError(
                "Zero valid molecule after checking when preprocessing ligands!!"
            )

        self.covalent_ligand = covalent_ligand
        self.template_docking = template_docking

        if reference_sdf_file_name is not None:
            self.reference_sdf_file_name = os.path.abspath(reference_sdf_file_name)
        else:
            self.reference_sdf_file_name = None

        self.core_atom_mapping_dict_list = core_atom_mapping_dict_list
        if self.core_atom_mapping_dict_list is None:
            self.core_atom_mapping_dict_list = [None] * self.num_ligands
        if len(self.core_atom_mapping_dict_list) != self.num_ligands:
            raise ValueError(
                "Number of user specified core atom mapping dicts does not match \
                    the number of input molecules!!"
            )

        if self.template_docking:
            if self.reference_sdf_file_name is None:
                raise ValueError(
                    "template docking mode specified without reference SDF file!!"
                )
        else:
            self.reference_sdf_file_name = None

        self.static_root = False
        if self.template_docking or self.covalent_ligand:
            self.static_root = True

        self.root_working_dir_name = os.path.abspath(working_dir_name)
        self.ligand_json_file_name = os.path.join(
            self.root_working_dir_name, 'ligands_unidock2.json'
        )

        self.n_cpu = os.cpu_count()
        if n_cpu:
            self.n_cpu = min(n_cpu, self.n_cpu)

        self.construct_ff = construct_ff
        self.atom_mapper_align = atom_mapper_align

        self.summary_ligand_info_json_file_name = os.path.join(
            self.root_working_dir_name, 'summary_ligand_info.json'
        )

    def generate_batch_ligand_topology(self):
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        raw_num_batches = self.n_cpu * 3
        batch_size = math.ceil(self.num_ligands / raw_num_batches)
        num_batches = math.ceil(self.num_ligands / batch_size)
        batch_ligand_topology_builder_results_list = [None] * num_batches

        pool = Pool(processes=self.n_cpu)
        for batch_idx, start_idx in enumerate(range(0, self.num_ligands, batch_size)):
            end_idx = min(start_idx + batch_size, self.num_ligands)
            batch_ligand_mol_list = self.ligand_mol_list[start_idx:end_idx]
            batch_core_atom_mapping_dict_list = self.core_atom_mapping_dict_list[start_idx:end_idx]

            working_dir_name = os.path.join(self.root_working_dir_name, f'ligand_batch_{batch_idx}')
            os.makedirs(working_dir_name, exist_ok=True)
            batch_ligand_sdf_file_name = os.path.join(working_dir_name, 'ligand_batch.sdf')
            with Chem.SDWriter(batch_ligand_sdf_file_name) as writer:
                for ligand_mol in batch_ligand_mol_list:
                    writer.write(ligand_mol)

            batch_ligand_topology_builder_results = pool.apply_async(
                    batch_topology_builder_process,
                    args=(
                        batch_ligand_sdf_file_name,
                        self.covalent_ligand,
                        self.template_docking,
                        self.reference_sdf_file_name,
                        batch_core_atom_mapping_dict_list,
                        working_dir_name,
                        self.construct_ff,
                        self.atom_mapper_align,
                    ),
            )

            batch_ligand_topology_builder_results_list[batch_idx] = batch_ligand_topology_builder_results

        self.total_ligand_info_dict_list = sum([p.get() for p in batch_ligand_topology_builder_results_list], [])
        pool.close()

        if len(self.total_ligand_info_dict_list) != self.num_ligands:
            raise ValueError(
                'Collected number of batch ligands does not equal to \
                    real total number of input ligands!!'
            )

    def get_summary_ligand_info_dict(self):
        self.summary_ligand_info_dict = {}
        for ligand_info_dict in self.total_ligand_info_dict_list:
            ligand_name = ligand_info_dict['ligand_name']
            self.summary_ligand_info_dict[ligand_name] = {'atoms': ligand_info_dict['atom_info'],
                                                          'torsions': ligand_info_dict['torsion_info'],
                                                          'root_atoms': ligand_info_dict['root_atom_idx'],
                                                          'fragment_atom_idx': ligand_info_dict['fragment_atom_idx']}

    def write_summary_ligand_info_json_file(self):
        with open(self.summary_ligand_info_json_file_name, 'w') as f:
            json.dump(self.summary_ligand_info_dict, f)
