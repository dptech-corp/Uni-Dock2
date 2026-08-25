import os
from shutil import copyfile

from unidock2.cli._arguments import add_config_arguments
from unidock2.cli._resolve import resolve_protein_prep_request


class CLICommand:
    """Perform protein preparation for large-batch docking.

    Values are resolved in this order: Pydantic defaults, YAML configuration,
    then explicitly supplied command-line arguments.
    """

    help = "Perform protein preparation"

    @staticmethod
    def add_arguments(parser):
        add_config_arguments(parser, "protein_prep")

    @staticmethod
    def run(args):
        from unidock2.io.get_temp_dir_prefix import get_temp_dir_prefix
        from unidock2.io.tempfile import TemporaryDirectory
        from unidock2.unidocktools.unidock_receptor_topology_builder import (
            UnidockReceptorTopologyBuilder,
        )

        request = resolve_protein_prep_request(args)
        temp_dir_prefix = os.path.join(
            request.root_temp_dir_name,
            get_temp_dir_prefix("protein_prep"),
        )

        with TemporaryDirectory(
            prefix=temp_dir_prefix,
            delete=request.remove_temp_dir,
        ) as temp_dir_name:
            receptor_builder = UnidockReceptorTopologyBuilder(
                request.receptor_file_name,
                prepared_hydrogen=request.config.preprocessing.preserve_receptor_hydrogen,
                covalent_residue_atom_info_list=(request.config.preprocessing.covalent_residue_atom_info_list),
                working_dir_name=temp_dir_name,
            )
            receptor_builder.generate_receptor_topology()
            copyfile(
                receptor_builder.receptor_parameterized_dms_file_name,
                request.receptor_dms_file_name,
            )
