import os

from unidock2.cli._arguments import add_config_arguments
from unidock2.cli._resolve import resolve_prepare_ligands_request
from unidock2.io.ud2lig import prep_from_config, write_ud2lig


class CLICommand:
    """Prepare ligands into a reusable UD2LIG directory.

    ``-l`` accepts a single SDF file or a directory of SDF files. ``-lb`` is
    unchanged. A UD2LIG directory cannot be prepared again.

    Values are resolved in this order: Pydantic defaults, YAML configuration,
    then explicitly supplied command-line arguments.
    """

    help = "Prepare ligands into a reusable UD2LIG directory"

    @staticmethod
    def add_arguments(parser):
        add_config_arguments(parser, "prepare_ligands")
        parser.add_argument(
            "-o",
            "--output_ud2lig_dir",
            required=True,
            dest="output_ud2lig_dir",
            help="Output UD2LIG directory",
        )

    @staticmethod
    def run(args):
        from unidock2.io.get_temp_dir_prefix import get_temp_dir_prefix
        from unidock2.io.tempfile import TemporaryDirectory
        from unidock2.unidocktools.unidock_ligand_topology_builder import (
            UnidockLigandTopologyBuilder,
        )

        request = resolve_prepare_ligands_request(args)
        os.makedirs(request.root_temp_dir_name, exist_ok=True)
        temp_dir_prefix = os.path.join(
            request.root_temp_dir_name,
            get_temp_dir_prefix("prepare_ligands"),
        )

        with TemporaryDirectory(
            prefix=temp_dir_prefix,
            delete=request.remove_temp_dir,
        ) as temp_dir_name:
            ligand_builder = UnidockLigandTopologyBuilder(
                list(request.ligand_sdf_file_name_list),
                n_cpu=request.config.hardware.n_cpu,
                working_dir_name=temp_dir_name,
                construct_ff=request.config.preprocessing.construct_ff,
            )
            ligand_builder.generate_batch_ligand_topology()
            ligand_builder.get_summary_ligand_info_dict()
            write_ud2lig(
                request.output_ud2lig_dir,
                ligand_builder.summary_ligand_info_dict,
                ligand_builder.ligand_mol_list,
                prep_from_config(request.config),
            )
        print(f"UD2LIG directory written to: {request.output_ud2lig_dir}")
