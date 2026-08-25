import json
import os
from typing import Any, Dict, List, Optional, Tuple

from unidock2._engine import build_engine_request, dump_engine_request
from unidock2.config import UnidockConfig


class _UnsetType:
    def __repr__(self):
        return "UNSET"


UNSET = _UnsetType()


def _write_engine_checkpoints(engine_request, working_dir):
    """Write the legacy topology payload and the replayable engine request."""
    with open(
        os.path.join(working_dir, "ud2_engine_inputs.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(engine_request["molecules"], file, allow_nan=False)

    dump_engine_request(
        engine_request,
        os.path.join(working_dir, "ud2_engine_request.json"),
    )


class UnidockProtocolRunner:
    """Run docking from typed configuration or the compatible legacy API."""

    def __init__(
        self,
        receptor_file_name: str,
        ligand_sdf_file_name_list: List[str],
        target_center: Tuple[float, float, float],
        box_size: Tuple[float, float, float] = UNSET,
        ligand_json_file_name: str = None,
        template_docking: bool = UNSET,
        reference_sdf_file_name: Optional[str] = UNSET,
        compute_center: bool = UNSET,
        core_atom_mapping_dict_list: Optional[List[Optional[Dict[int, int]]]] = UNSET,
        covalent_ligand: bool = UNSET,
        covalent_residue_atom_info_list: Optional[List[Dict[str, Any]]] = UNSET,
        construct_ff: bool = UNSET,
        atom_mapper_align: bool = False,
        preserve_receptor_hydrogen: bool = UNSET,
        working_dir_name: str = ".",
        docking_pose_sdf_file_name: str = UNSET,
        n_cpu: Optional[int] = UNSET,
        gpu_device_id: int = UNSET,
        task: str = UNSET,
        search_mode: str = UNSET,
        exhaustiveness: int = UNSET,
        randomize: bool = UNSET,
        mc_steps: int = UNSET,
        opt_steps: int = UNSET,
        refine_steps: int = UNSET,
        num_pose: int = UNSET,
        rmsd_limit: float = UNSET,
        energy_range: float = UNSET,
        seed: int = UNSET,
        use_tor_lib: bool = UNSET,
        energy_decomp: bool = UNSET,
        engine_checkpoint: bool = UNSET,
        bias: str = UNSET,
        bias_k: float = UNSET,
        max_gpu_memory: int = UNSET,
        **config_overrides: Any,
    ) -> None:
        """Adapt the historical constructor to the typed configuration path.

        Existing parameter names and positional order are retained. ``UNSET``
        distinguishes omitted values from explicit values, allowing all business
        defaults to come from ``UnidockConfig``.
        """
        config = UnidockConfig()
        local_values = locals()
        overrides = dict(config_overrides)

        for field_name in config.protocol_field_names():
            value = local_values.get(field_name, UNSET)
            if value is not UNSET:
                overrides[field_name] = value

        overrides["center"] = list(target_center)
        if docking_pose_sdf_file_name is not UNSET:
            overrides["output_docking_pose_sdf_file_name"] = docking_pose_sdf_file_name
        config = config.with_overrides(**overrides)

        self._initialize(
            receptor_file_name=receptor_file_name,
            ligand_sdf_file_name_list=ligand_sdf_file_name_list,
            target_center=tuple(config.required.center),
            working_dir_name=working_dir_name,
            ligand_json_file_name=ligand_json_file_name,
            atom_mapper_align=atom_mapper_align,
            config=config,
        )

    @classmethod
    def from_config(
        cls,
        receptor_file_name: str,
        ligand_sdf_file_name_list,
        target_center,
        config: Optional[UnidockConfig] = None,
        working_dir_name: str = ".",
        docking_pose_sdf_file_name: Optional[str] = None,
        ligand_json_file_name: Optional[str] = None,
        atom_mapper_align: bool = False,
    ):
        """Create a runner directly from the canonical typed configuration."""
        if config is None:
            config = UnidockConfig()
        elif not isinstance(config, UnidockConfig):
            config = UnidockConfig.model_validate(config)

        overrides = {"center": list(target_center)}
        if docking_pose_sdf_file_name is not None:
            overrides["output_docking_pose_sdf_file_name"] = docking_pose_sdf_file_name
        config = config.with_overrides(**overrides)

        runner = cls.__new__(cls)
        runner._initialize(
            receptor_file_name=receptor_file_name,
            ligand_sdf_file_name_list=ligand_sdf_file_name_list,
            target_center=tuple(config.required.center),
            working_dir_name=working_dir_name,
            ligand_json_file_name=ligand_json_file_name,
            atom_mapper_align=atom_mapper_align,
            config=config,
        )
        return runner

    def _initialize(
        self,
        receptor_file_name,
        ligand_sdf_file_name_list,
        target_center,
        working_dir_name,
        ligand_json_file_name,
        atom_mapper_align,
        config,
    ):
        self.config = config
        for field_name, value in config.to_protocol_kwargs().items():
            setattr(self, field_name, value)

        self.receptor_file_name = os.path.abspath(receptor_file_name)
        self.ligand_sdf_file_name_list = [os.path.abspath(file_name) for file_name in ligand_sdf_file_name_list]
        self.ligand_json_file_name = (
            os.path.abspath(ligand_json_file_name) if ligand_json_file_name is not None else None
        )
        self.target_center = target_center
        self.atom_mapper_align = atom_mapper_align
        self.reference_sdf_file_name = (
            os.path.abspath(self.reference_sdf_file_name) if self.reference_sdf_file_name else None
        )
        self.working_dir_name = os.path.abspath(working_dir_name)
        self.unidock2_output_dir_name = os.path.join(
            self.working_dir_name,
            "unidock2_output",
        )
        self.docking_pose_sdf_file_name = os.path.abspath(self.output_docking_pose_sdf_file_name)
        os.makedirs(self.unidock2_output_dir_name, exist_ok=True)

        self.core_atom_mapping_dict_list = (
            [
                {int(key): int(value) for key, value in mapping.items()} if mapping else None
                for mapping in self.core_atom_mapping_dict_list
            ]
            if self.core_atom_mapping_dict_list
            else None
        )

        if self.template_docking and self.reference_sdf_file_name and self.compute_center:
            self.target_center = self._center_from_sdf(self.reference_sdf_file_name)

        if self.covalent_ligand and self.compute_center:
            self.target_center = self._center_from_sdf(self.ligand_sdf_file_name_list[0])

        print(f"Target Center for Current Docking: {self.target_center}")

        if self.ligand_json_file_name:
            with open(self.ligand_json_file_name, encoding="utf-8") as ligand_json_file:
                self.specified_ligand_info_dict = json.load(ligand_json_file)
        else:
            self.specified_ligand_info_dict = None

        if self.receptor_file_name.split(".")[-1] == "json":
            with open(self.receptor_file_name, encoding="utf-8") as receptor_file:
                self.specified_receptor_info_dict = json.load(receptor_file)
        else:
            self.specified_receptor_info_dict = None

    @staticmethod
    def _center_from_sdf(sdf_file_name):
        from rdkit import Chem

        from unidock2.ligand_topology import utils

        molecule = Chem.SDMolSupplier(sdf_file_name, removeHs=True)[0]
        return tuple(utils.calculate_center_of_mass(molecule))

    def _current_config(self):
        overrides = {
            field_name: getattr(self, field_name)
            for field_name in self.config.protocol_field_names()
            if hasattr(self, field_name)
        }
        return self.config.with_overrides(**overrides)

    def run_unidock_protocol(self) -> str:
        from unidock2._engine import pipeline
        from unidock2.unidocktools.unidock_ligand_pose_writer import (
            UnidockLigandPoseWriter,
        )
        from unidock2.unidocktools.unidock_ligand_topology_builder import (
            UnidockLigandTopologyBuilder,
        )
        from unidock2.unidocktools.unidock_receptor_topology_builder import (
            UnidockReceptorTopologyBuilder,
        )

        if self.specified_receptor_info_dict:
            print("Using specified receptor info dict...")
            receptor_atom_info_list = self.specified_receptor_info_dict["receptor"]
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

        use_specified_ligand_info = bool(self.specified_ligand_info_dict)
        if use_specified_ligand_info:
            print("Using specified ligand info dict...")

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
        if use_specified_ligand_info:
            ligand_info_dict = self.specified_ligand_info_dict
        else:
            ligand_builder.generate_batch_ligand_topology()
            ligand_builder.get_summary_ligand_info_dict()
            ligand_info_dict = ligand_builder.summary_ligand_info_dict

        engine_request = build_engine_request(
            self._current_config(),
            target_center=self.target_center,
            output_dir=self.unidock2_output_dir_name,
            receptor=receptor_atom_info_list,
            ligands=ligand_info_dict,
        )

        if self.engine_checkpoint:
            _write_engine_checkpoints(engine_request, self.working_dir_name)

        pipeline.run(engine_request)

        pose_json_files = [
            os.path.join(self.unidock2_output_dir_name, file_name)
            for file_name in os.listdir(self.unidock2_output_dir_name)
            if file_name.endswith(".json")
        ]
        pose_writer = UnidockLigandPoseWriter(
            ligand_builder.ligand_mol_list,
            pose_json_files,
            covalent_ligand=self.covalent_ligand,
            energy_decomp=self.energy_decomp,
            docking_pose_sdf_file_name=self.docking_pose_sdf_file_name,
        )
        pose_writer.generate_docking_pose_sdf()

        return self.docking_pose_sdf_file_name
