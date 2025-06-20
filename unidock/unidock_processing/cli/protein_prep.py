class CLICommand:
    """Perform protein preparation only for large batch docking.

    Note: Required arguments should be passed by either command line arguments or YAML Required options.
    If both sides are specified, the values passed in the command line arguments will overide the values passed in YAML file.

    Example input YAML configurations:

    Required:
        receptor: 1G9V_protein_water_cleaned.pdb
    Preprocessing:
        covalent_residue_atom_info_list: null
        preserve_receptor_hydrogen: false
        temp_dir_name: str = /tmp
        output_receptor_dms_file_name = receptor_parameterized.dms

    """

    def add_arguments(parser):
        parser.add_argument(
            '-r',
            '--receptor',
            default=None,
            help='Receptor structure file in PDB or DMS format',
        )

        parser.add_argument(
            '-o',
            '--output_receptor_dms_file_name',
            default='receptor_parameterized.dms',
            help='Output receptor DMS file name',
        )

        parser.add_argument(
            '-cf',
            '--configurations',
            default=None,
            help='Uni-Dock2 configuration YAML file recording all other options',
        )

    def run(args):
        import os
        from unidock_processing.io.yaml import read_unidock_params_from_yaml
        from unidock_processing.io.get_temp_dir_prefix import get_temp_dir_prefix
        from unidock_processing.io.tempfile import TemporaryDirectory
        from unidock_processing.unidocktools.unidock_receptor_topology_builder import (
        UnidockReceptorTopologyBuilder,
        )

        ## Read all arguments from user input YAML first
        kwargs_dict = {}
        if args.configurations:
            extra_params = read_unidock_params_from_yaml(args.configurations)
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

        ## Specify receptor DMS file name
        kwargs_receptor_dms_file_name = kwargs_dict.pop('output_receptor_dms_file_name', None)
        if args.output_receptor_dms_file_name != 'receptor_parameterized.dms':
            receptor_dms_file_name = args.output_receptor_dms_file_name
        else:
            receptor_dms_file_name = kwargs_receptor_dms_file_name

        receptor_dms_file_name = os.path.abspath(receptor_dms_file_name)

        ## Prepare temp dir
        root_temp_dir_name = os.path.abspath(kwargs_dict.pop('temp_dir_name', None))
        temp_dir_prefix = os.path.join(
            root_temp_dir_name, get_temp_dir_prefix('protein_prep')
        )

        if root_temp_dir_name == '/tmp':
            remove_temp_dir = True
        else:
            remove_temp_dir = False

        ## Run receptor preparation
        with TemporaryDirectory(prefix=temp_dir_prefix, delete=remove_temp_dir) as temp_dir_name:
            unidock_receptor_topology_builder = UnidockReceptorTopologyBuilder(
                receptor_file_name,
                prepared_hydrogen=kwargs_dict['preserve_receptor_hydrogen'],
                covalent_residue_atom_info_list=kwargs_dict['covalent_residue_atom_info_list'],
                working_dir_name=temp_dir_name,
            )

            unidock_receptor_topology_builder.generate_receptor_topology()
            os.system(f'cp {unidock_receptor_topology_builder.receptor_parameterized_dms_file_name} {receptor_dms_file_name}')
