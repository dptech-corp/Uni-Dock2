class CLICommand:
    """Perform whole docking protocol.

    Note: Required arguments should be passed by either command line arguments or YAML Required options.
    If both sides are specified,
    the values passed in the command line arguments will overide the values passed in YAML file.

    Example input YAML configurations:

    Required:
        receptor: 1G9V_protein_water_cleaned.pdb
        ligand: ligand_prepared.sdf
        ligand_batch: null
        center: [5.122, 18.327, 37.332]
    Advanced:
        exhaustiveness: 512
        randomize: true
        mc_steps: 20
        opt_steps: -1
        refine_steps: 5
        num_pose: 10
        rmsd_limit: 1.0
        energy_range: 3.0
        seed: 12345
        use_tor_lib: false
    Hardware:
        gpu_device_id: 0
    Settings:
        size: [30.0, 30.0, 30.0]
        task: screen
        search_mode: balance
    Preprocessing:
        template_docking: false
        reference_sdf_file_name: null
        core_atom_mapping_dict_list: null
        covalent_ligand: false
        covalent_residue_atom_info_list: null
        preserve_receptor_hydrogen: false
        temp_dir_name: /tmp
        output_docking_pose_sdf_file_name: unidock2_pose.sdf

    """

    def add_arguments(parser):
        parser.add_argument(
            '-r',
            '--receptor',
            default=None,
            help='Receptor structure file in PDB or DMS format',
        )

        parser.add_argument(
            '-l',
            '--ligand',
            default=None,
            help='Single ligand structure file in SDF format',
        )

        parser.add_argument(
            '-lb',
            '--ligand_batch',
            default=None,
            help='Recorded batch text file of ligand SDF file path',
        )

        parser.add_argument(
            '-c',
            '--center',
            nargs=3,
            type=float,
            metavar=('center_x', 'center_y', 'center_z'),
            default=[0.0, 0.0, 0.0],
            help='Docking box center coordinates',
        )

        parser.add_argument(
            '-o',
            '--output_docking_pose_sdf_file_name',
            default='unidock2_pose.sdf',
            help='Output docking pose SDF file name',
        )

        parser.add_argument(
            '-cf',
            '--configurations',
            default=None,
            help='Uni-Dock2 configuration YAML file recording all other options',
        )

    def run(args):
        import os
        from unidock_processing.io.yaml import UnidockConfig, read_unidock_params_from_yaml
        from unidock_processing.io.get_temp_dir_prefix import get_temp_dir_prefix
        from unidock_processing.io.tempfile import TemporaryDirectory
        from unidock_processing.unidocktools.unidock_protocol_runner import (
            UnidockProtocolRunner,
        )

        ## Read all arguments from user input YAML first
        kwargs_dict = {}
        if args.configurations:
            extra_params = read_unidock_params_from_yaml(args.configurations)
            kwargs_dict = extra_params.to_protocol_kwargs()
        else:
            extra_params = UnidockConfig()
            kwargs_dict = extra_params.to_protocol_kwargs()

        print(kwargs_dict)

        ## Parse receptor input
        kwargs_receptor_file_name = kwargs_dict.pop('receptor', None)
        if kwargs_receptor_file_name is not None:
            kwargs_receptor_file_name = os.path.abspath(kwargs_receptor_file_name)

        if args.receptor is not None:
            receptor_file_name = os.path.abspath(args.receptor)
        else:
            receptor_file_name = kwargs_receptor_file_name

        if receptor_file_name is None:
            raise ValueError('Receptor file name not specified !')

        ## Parse single ligand sdf file input
        kwargs_ligand_sdf_file_name = kwargs_dict.pop('ligand', None)
        if kwargs_ligand_sdf_file_name is not None:
            kwargs_ligand_sdf_file_name = os.path.abspath(kwargs_ligand_sdf_file_name)

        if args.ligand is not None:
            ligand_sdf_file_name = os.path.abspath(args.ligand)
        else:
            ligand_sdf_file_name = kwargs_ligand_sdf_file_name

        ## Parse ligand batch file input
        kwargs_ligand_batch_file_name = kwargs_dict.pop('ligand_batch', None)
        if kwargs_ligand_batch_file_name is not None:
            kwargs_ligand_batch_file_name = os.path.abspath(kwargs_ligand_batch_file_name)

        if args.ligand_batch is not None:
            ligand_batch_file_name = os.path.abspath(args.ligand_batch)
        else:
            ligand_batch_file_name = kwargs_ligand_batch_file_name

        if ligand_sdf_file_name is None and ligand_batch_file_name is None:
            raise ValueError('Ligand SDF file input not found !')

        ## Summarize ligand input sdf files
        if ligand_batch_file_name:
            with open(ligand_batch_file_name, 'r') as ligand_batch_file:
                ligand_batch_line_list = ligand_batch_file.readlines()

            batch_ligand_sdf_file_name_list = []
            for ligand_batch_line in ligand_batch_line_list:
                batch_ligand_sdf_file_name = ligand_batch_line.strip()
                if len(batch_ligand_sdf_file_name) != 0:
                    batch_ligand_sdf_file_name_list.append(
                        os.path.abspath(batch_ligand_sdf_file_name)
                    )
        else:
            batch_ligand_sdf_file_name_list = []

        if ligand_sdf_file_name:
            total_ligand_sdf_file_name_list = [
                ligand_sdf_file_name
            ] + batch_ligand_sdf_file_name_list
        else:
            total_ligand_sdf_file_name_list = batch_ligand_sdf_file_name_list

        if len(total_ligand_sdf_file_name_list) == 0:
            raise ValueError('Ligand SDF file input not found !!')

        ## Parse docking center input
        kwargs_center = kwargs_dict.pop('center', None)
        if args.center != (0.0, 0.0, 0.0):
            docking_center = args.center
        else:
            docking_center = kwargs_center

        ## Specify docking pose SDF file name
        kwargs_docking_pose_sdf_file_name = kwargs_dict.pop('output_docking_pose_sdf_file_name', None)
        if args.output_docking_pose_sdf_file_name != 'unidock2_pose.sdf':
            docking_pose_sdf_file_name = args.output_docking_pose_sdf_file_name
        else:
            docking_pose_sdf_file_name = kwargs_docking_pose_sdf_file_name

        docking_pose_sdf_file_name = os.path.abspath(docking_pose_sdf_file_name)
        _ = kwargs_dict.pop('output_receptor_dms_file_name', None)

        ## Prepare temp dir
        root_temp_dir_name = os.path.abspath(kwargs_dict.pop('temp_dir_name', None))
        temp_dir_prefix = os.path.join(
            root_temp_dir_name, get_temp_dir_prefix('docking')
        )

        if root_temp_dir_name == '/tmp':
            remove_temp_dir = True
        else:
            remove_temp_dir = False

        ## Run docking protocol
        with TemporaryDirectory(prefix=temp_dir_prefix, delete=remove_temp_dir) as temp_dir_name:
            docking_runner = UnidockProtocolRunner(
                receptor_file_name=receptor_file_name,
                ligand_sdf_file_name_list=total_ligand_sdf_file_name_list,
                target_center=tuple(docking_center),
                working_dir_name=temp_dir_name,
                docking_pose_sdf_file_name=docking_pose_sdf_file_name,
                **kwargs_dict,
            )

            docking_runner.run_unidock_protocol()
