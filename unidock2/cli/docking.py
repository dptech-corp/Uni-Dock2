import os

from unidock2.cli._arguments import add_config_arguments
from unidock2.cli._resolve import resolve_docking_request


class CLICommand:
    """Perform the complete docking protocol.

    Values are resolved in this order: Pydantic defaults, YAML configuration,
    then explicitly supplied command-line arguments.
    """

    @staticmethod
    def add_arguments(parser):
        add_config_arguments(parser, "docking")

    @staticmethod
    def run(args):
        from unidock2.io.get_temp_dir_prefix import get_temp_dir_prefix
        from unidock2.io.tempfile import TemporaryDirectory
        from unidock2.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )

        request = resolve_docking_request(args)

        os.makedirs(request.root_temp_dir_name, exist_ok=True)
        temp_dir_prefix = os.path.join(
            request.root_temp_dir_name,
            get_temp_dir_prefix("docking"),
        )

        with TemporaryDirectory(
            prefix=temp_dir_prefix,
            delete=request.remove_temp_dir,
        ) as temp_dir_name:
            docking_runner = UnidockProtocolRunner.from_config(
                receptor_file_name=request.receptor_file_name,
                ligand_sdf_file_name_list=request.ligand_sdf_file_name_list,
                target_center=request.target_center,
                working_dir_name=temp_dir_name,
                docking_pose_sdf_file_name=request.docking_pose_sdf_file_name,
                config=request.config,
            )
            docking_runner.run_unidock_protocol()
