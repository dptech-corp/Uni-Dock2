from typing import List, Optional, Tuple, Dict, Any
import os
import json

from rdkit import Chem

from unidock_engine.api.python import pipeline
from unidock_processing.unidocktools.unidock_receptor_topology_builder import (
    UnidockReceptorTopologyBuilder,
)
from unidock_processing.unidocktools.unidock_ligand_topology_builder import (
    UnidockLigandTopologyBuilder,
)
from unidock_processing.unidocktools.unidock_ligand_pose_writer import (
    UnidockLigandPoseWriter,
)
from unidock_processing.ligand_topology import utils

class UnidockProtocolRunner(object):
    def __init__(
        self,
        receptor_file_name: str,
        ligand_sdf_file_name_list: List[str],
        target_center: Tuple[float, float, float],
        box_size: Tuple[float, float, float] = (30.0, 30.0, 30.0),
        ligand_json_file_name: str = None,
        template_docking: bool = False,
        reference_sdf_file_name: Optional[str] = None,
        compute_center: bool = True,
        core_atom_mapping_dict_list: Optional[List[Optional[Dict[int, int]]]] = None,
        covalent_ligand: bool = False,
        covalent_residue_atom_info_list: Optional[List[Dict[str, Any]]] = None,
        construct_ff: bool = False,
        atom_mapper_align: bool = False,
        preserve_receptor_hydrogen: bool = False,
        working_dir_name: str = '.',
        docking_pose_sdf_file_name: str = 'unidock2_pose.sdf',
        n_cpu: Optional[int] = None,
        gpu_device_id: int = 0,
        task: str = 'screen',
        search_mode: str = 'balance',
        exhaustiveness: int = 512,
        randomize: bool = True,
        mc_steps: int = 40,
        opt_steps: int = -1,
        refine_steps: int = 5,
        num_pose: int = 10,
        rmsd_limit: float = 1.0,
        energy_range: float = 5.0,
        seed: int = 1234567,
        use_tor_lib: bool = False,
        engine_checkpoint: bool = False,
    ) -> None:

        self.receptor_file_name = os.path.abspath(receptor_file_name)
        self.ligand_sdf_file_name_list = [os.path.abspath(f) for f in ligand_sdf_file_name_list]

        if ligand_json_file_name is not None:
            self.ligand_json_file_name = os.path.abspath(ligand_json_file_name)
        else:
            self.ligand_json_file_name = None

        self.target_center = target_center
        self.template_docking = template_docking
        self.reference_sdf_file_name = os.path.abspath(reference_sdf_file_name) if reference_sdf_file_name else None
        self.compute_center = compute_center
        self.covalent_ligand = covalent_ligand
        self.covalent_residue_atom_info_list = covalent_residue_atom_info_list
        self.construct_ff = construct_ff
        self.atom_mapper_align = atom_mapper_align
        self.preserve_receptor_hydrogen = preserve_receptor_hydrogen
        self.box_size = box_size
        self.n_cpu = n_cpu
        self.gpu_device_id = gpu_device_id
        self.task = task
        self.search_mode = search_mode
        self.exhaustiveness = exhaustiveness
        self.randomize = randomize
        self.mc_steps = mc_steps
        self.opt_steps = opt_steps
        self.refine_steps = refine_steps
        self.num_pose = num_pose
        self.rmsd_limit = rmsd_limit
        self.energy_range = energy_range
        self.seed = seed
        self.use_tor_lib = use_tor_lib
        self.engine_checkpoint = engine_checkpoint
        self.working_dir_name = os.path.abspath(working_dir_name)
        self.unidock2_output_dir_name = os.path.join(self.working_dir_name, 'unidock2_output')
        self.docking_pose_sdf_file_name = os.path.abspath(docking_pose_sdf_file_name)
        os.makedirs(self.unidock2_output_dir_name, exist_ok=True)

        self.core_atom_mapping_dict_list = [
            {int(k): int(v) for k, v in d.items()} if d else None
            for d in core_atom_mapping_dict_list
        ] if core_atom_mapping_dict_list else None

        if self.template_docking and self.reference_sdf_file_name and self.compute_center:
            reference_mol = Chem.SDMolSupplier(self.reference_sdf_file_name, removeHs=True)[0]
            self.target_center = tuple(utils.calculate_center_of_mass(reference_mol))

        if self.covalent_ligand and self.compute_center:
            ligand_mol = Chem.SDMolSupplier(self.ligand_sdf_file_name_list[0], removeHs=True)[0]
            self.target_center = tuple(utils.calculate_center_of_mass(ligand_mol))

        print(f'Target Center for Current Docking: {self.target_center}')

        if self.ligand_json_file_name:
            with open(self.ligand_json_file_name) as ligand_json_file:
                self.specified_ligand_info_dict = json.load(ligand_json_file)
        else:
            self.specified_ligand_info_dict = None

        if self.receptor_file_name.split('.')[-1] == 'json':
            with open(self.receptor_file_name) as receptor_file:
                self.specified_receptor_info_dict = json.load(receptor_file)
        else:
            self.specified_receptor_info_dict = None

    def run_unidock_protocol(self) -> str:
        # Prepare receptor
        if self.specified_receptor_info_dict:
            print('Using specified receptor info dict...')
            receptor_atom_info_list = self.specified_receptor_info_dict['receptor']
        else:
            receptor_builder = UnidockReceptorTopologyBuilder(
                self.receptor_file_name,
                prepared_hydrogen=self.preserve_receptor_hydrogen,
                covalent_residue_atom_info_list=self.covalent_residue_atom_info_list,
                working_dir_name=self.working_dir_name,
            )
            receptor_builder.generate_receptor_topology()
            receptor_builder.analyze_receptor_topology()
            receptor_builder.get_summary_receptor_info()
            receptor_atom_info_list = receptor_builder.atom_info_nested_list

        # Prepare ligands
        if self.specified_ligand_info_dict:
            print('Using specified ligand info dict...')
            ligand_info_dict = self.specified_ligand_info_dict
            ligand_builder = UnidockLigandTopologyBuilder(
                self.ligand_sdf_file_name_list,
                covalent_ligand=self.covalent_ligand,
                template_docking=self.template_docking,
                reference_sdf_file_name=self.reference_sdf_file_name,
                core_atom_mapping_dict_list=self.core_atom_mapping_dict_list,
                n_cpu=self.n_cpu,
                working_dir_name=self.working_dir_name,
                construct_ff=self.construct_ff,
                atom_mapper_align=self.atom_mapper_align,
            )
        else:
            ligand_builder = UnidockLigandTopologyBuilder(
                self.ligand_sdf_file_name_list,
                covalent_ligand=self.covalent_ligand,
                template_docking=self.template_docking,
                reference_sdf_file_name=self.reference_sdf_file_name,
                core_atom_mapping_dict_list=self.core_atom_mapping_dict_list,
                n_cpu=self.n_cpu,
                working_dir_name=self.working_dir_name,
                construct_ff=self.construct_ff,
                atom_mapper_align=self.atom_mapper_align,
            )
            ligand_builder.generate_batch_ligand_topology()
            ligand_builder.get_summary_ligand_info_dict()
            ligand_info_dict = ligand_builder.summary_ligand_info_dict

        if self.engine_checkpoint:
            with open(os.path.join(self.working_dir_name, 'ud2_engine_inputs.json'), 'w') as f:
                json.dump({
                    'receptor': receptor_atom_info_list,
                    **ligand_info_dict
                }, f)

        # Instantiate and configure the docking pipeline
        docking_pipeline = pipeline.DockingPipeline(
            output_dir=self.unidock2_output_dir_name,
            center_x=self.target_center[0],
            center_y=self.target_center[1],
            center_z=self.target_center[2],
            size_x=self.box_size[0],
            size_y=self.box_size[1],
            size_z=self.box_size[2],
            task=self.task,
            search_mode=self.search_mode,
            exhaustiveness=self.exhaustiveness,
            randomize=self.randomize,
            mc_steps=self.mc_steps,
            opt_steps=self.opt_steps,
            refine_steps=self.refine_steps,
            num_pose=self.num_pose,
            rmsd_limit=self.rmsd_limit,
            energy_range=self.energy_range,
            seed=self.seed,
            use_tor_lib=self.use_tor_lib,
            constraint_docking=self.template_docking or self.covalent_ligand,
            gpu_device_id=self.gpu_device_id
        )

        docking_pipeline.set_receptor(receptor_atom_info_list)
        docking_pipeline.add_ligands(ligand_info_dict)

        docking_pipeline.run()

        # Process and write output poses
        pose_json_files = [
            os.path.join(self.unidock2_output_dir_name, f)
            for f in os.listdir(self.unidock2_output_dir_name) if f.endswith('.json')
        ]
        pose_writer = UnidockLigandPoseWriter(
            ligand_builder.ligand_mol_list,
            pose_json_files,
            covalent_ligand=self.covalent_ligand,
            docking_pose_sdf_file_name=self.docking_pose_sdf_file_name,
        )
        pose_writer.generate_docking_pose_sdf()

        return self.docking_pose_sdf_file_name
