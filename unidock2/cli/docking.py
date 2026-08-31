from unidock2.cli._arguments import add_config_arguments
from unidock2.cli._resolve import resolve_docking_request
from unidock2.io.yaml import DEFAULT_CONFIG_FILE_NAME, dump_default_config_yaml


class CLICommand:
    """Perform the complete docking protocol.

    Receptor ``-r`` accepts PDB or DMS. A DMS file skips protein preparation.

    Ligand ``-l`` accepts a single SDF file, a directory of SDF files, or a
    UD2LIG directory (``manifest.json`` with magic ``ud2lig``). ``-lb`` is
    unchanged and cannot be combined with a UD2LIG directory.

    After ligand preparation, docking writes a reusable UD2LIG directory next
    to ``-o`` / ``--output_sdf`` by default. Disable with ``--no-engine_checkpoint``.
    Intermediate receptor DMS in the working directory is always
    ``receptor_parameterized.dms``.

    Intermediate files go to one directory per run under ``unidock2_temp``
    beside the output SDF. A successful run removes it, a failed run keeps it,
    and ``--keep_workdir`` always keeps it.

    Values are resolved in this order: Pydantic defaults, YAML configuration,
    then explicitly supplied command-line arguments.
    """

    help = "Perform docking"

    @staticmethod
    def add_arguments(parser):
        add_config_arguments(parser, "docking")
        parser.add_argument(
            "--dump_config",
            "--dump-config",
            nargs="?",
            const=DEFAULT_CONFIG_FILE_NAME,
            default=None,
            metavar="FILE",
            help=(f"Write an annotated default YAML configuration and exit (default file: {DEFAULT_CONFIG_FILE_NAME})"),
        )

    @staticmethod
    def run(args):
        if getattr(args, "dump_config", None) is not None:
            output_path = dump_default_config_yaml(args.dump_config)
            print(f"Default configuration written to: {output_path}")
            return output_path

        from unidock2.io.workdir import run_workdir
        from unidock2.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )

        request = resolve_docking_request(args)

        with run_workdir(
            request.workdir_root,
            "docking",
            keep=request.keep_workdir,
        ) as working_dir_name:
            docking_runner = UnidockProtocolRunner.from_config(
                receptor_file_name=request.receptor_file_name,
                ligand_sdf_file_name_list=request.ligand_sdf_file_name_list,
                target_center=request.target_center,
                working_dir_name=working_dir_name,
                docking_pose_sdf_file_name=request.docking_pose_sdf_file_name,
                config=request.config,
                ud2lig_dir=request.ud2lig_dir,
            )
            docking_runner.run_unidock_protocol()
