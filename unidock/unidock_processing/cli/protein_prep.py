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
        working_dir_name: .
    """

    def add_arguments(parser):
        parser.add_argument(
            '-r',
            '--receptor',
            default=None,
            help='Receptor structure file in PDB or DMS format',
        )

        parser.add_argument(
            '-cf',
            '--configurations',
            default=None,
            help='Uni-Dock2 configuration YAML file recording all other options',
        )

    def run(args):
        import os
        from unidock_processing.io import read_unidock_params_from_yaml
        from unidock_processing.unidocktools.unidock_receptor_topology_builder import (
        UnidockReceptorTopologyBuilder,
        )

        kwargs_dict = {}
        if args.configurations:
            extra_params = read_unidock_params_from_yaml(args.configurations)
            kwargs_dict = extra_params.to_protocol_kwargs()
        print(kwargs_dict)

        kwargs_receptor_file_name = kwargs_dict.pop('receptor', None)
        if kwargs_receptor_file_name is not None:
            kwargs_receptor_file_name = os.path.abspath(kwargs_receptor_file_name)

        if args.receptor is not None:
            receptor_file_name = os.path.abspath(args.receptor)
        else:
            receptor_file_name = kwargs_receptor_file_name

        if receptor_file_name is None:
            raise ValueError('Receptor file name not specified !')

        unidock_receptor_topology_builder = UnidockReceptorTopologyBuilder(
            receptor_file_name,
            prepared_hydrogen=kwargs_dict['preserve_receptor_hydrogen'],
            covalent_residue_atom_info_list=kwargs_dict['covalent_residue_atom_info_list'],
            working_dir_name=kwargs_dict['working_dir_name'],
        )

        unidock_receptor_topology_builder.generate_receptor_topology()
