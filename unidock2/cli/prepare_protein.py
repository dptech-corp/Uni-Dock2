from shutil import copyfile

from unidock2.cli._arguments import add_config_arguments
from unidock2.cli._resolve import resolve_prepare_protein_request


class CLICommand:
    """Prepare a receptor structure for docking.

    A PDB input runs protein preparation and writes a DMS file. A DMS input is
    copied through as an already prepared receptor.

    Intermediate files go to one directory per run under ``unidock2_temp``
    beside the output DMS. A successful run removes it, a failed run keeps it,
    and ``--keep_workdir`` always keeps it.

    ``-o`` is required. This command does not read a YAML configuration file.
    """

    help = "Prepare a receptor (PDB or DMS) into a reusable DMS file"

    @staticmethod
    def add_arguments(parser):
        add_config_arguments(parser, "prepare_protein", with_config_file=False)
        parser.add_argument(
            "-o",
            "--output",
            required=True,
            dest="output_dms",
            metavar="DMS",
            help="Output receptor DMS file",
        )

    @staticmethod
    def run(args):
        from unidock2.io.workdir import run_workdir
        from unidock2.unidocktools.unidock_receptor_topology_builder import (
            UnidockReceptorTopologyBuilder,
        )

        request = resolve_prepare_protein_request(args)

        with run_workdir(
            request.workdir_root,
            "prepare_protein",
            keep=request.keep_workdir,
        ) as working_dir_name:
            receptor_builder = UnidockReceptorTopologyBuilder(
                request.receptor_file_name,
                prepared_hydrogen=request.config.preprocessing.preserve_receptor_hydrogen,
                covalent_residue_atom_info_list=(request.config.preprocessing.covalent_residue_atom_info_list),
                working_dir_name=working_dir_name,
            )
            receptor_builder.generate_receptor_topology()
            copyfile(
                receptor_builder.receptor_parameterized_dms_file_name,
                request.receptor_dms_file_name,
            )
        print(f"Receptor DMS written to: {request.receptor_dms_file_name}")
