import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unidock.unidock_processing.unidocktools.unidock_protocol_runner import UnidockProtocolRunner
from unidock import __version__

logo_description = r"""

    ██╗   ██╗██████╗ ██████╗ 
    ██║   ██║██╔══██╗╚════██╗
    ██║   ██║██║  ██║ █████╔╝
    ██║   ██║██║  ██║██╔═══╝ 
    ╚██████╔╝██████╔╝███████╗
     ╚═════╝ ╚═════╝ ╚══════╝

    DP Technology Docking Toolkit

"""

def main():
    print(logo_description)
    parser = argparse.ArgumentParser(
        prog='unidock2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-r', '--receptor', required=True,
                        help='Receptor structure file in PDB or DMS format')

    parser.add_argument('-l', '--ligand',
                        default=None,
                        help='Single ligand structure file in SDF format')

    parser.add_argument('-lb', '--ligand_batch',
                        default=None,
                        help='Recorded batch text file of ligand SDF file path')

    parser.add_argument('-c', '--center', nargs=3, type=float, metavar=('center_x', 'center_y', 'center_z'),
                        default=[0.0, 0.0, 0.0],
                        help='Docking box center coordinates')

    parser.add_argument('-cf', '--configurations',
                        default=None,
                        help='Uni-Dock2 configuration file recording all other options')

    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s {__version__}',
                        help='Show program version')

    args = parser.parse_args()

    receptor_file_name = os.path.abspath(args.receptor)

    if args.ligand:
        ligand_sdf_file_name = os.path.abspath(args.ligand)
    else:
        ligand_sdf_file_name = None

    if args.ligand_batch:
        with open(args.ligand_batch, 'r') as ligand_batch_file:
            ligand_batch_line_list = ligand_batch_file.readlines()

        batch_ligand_sdf_file_name_list = []
        for ligand_batch_line in ligand_batch_line_list:
            batch_ligand_sdf_file_name = ligand_batch_line.strip()
            if len(batch_ligand_sdf_file_name) != 0:
                batch_ligand_sdf_file_name_list.append(os.path.abspath(batch_ligand_sdf_file_name))
    else:
        batch_ligand_sdf_file_name_list = []

    if ligand_sdf_file_name:
        total_ligand_sdf_file_name_list = [ligand_sdf_file_name] + batch_ligand_sdf_file_name_list
    else:
        total_ligand_sdf_file_name_list = batch_ligand_sdf_file_name_list

    if len(total_ligand_sdf_file_name_list) == 0:
        raise ValueError('Ligand SDF file input not found !!')

    docking_runner = UnidockProtocolRunner(
        receptor_file_name=receptor_file_name,
        ligand_sdf_file_name_list=total_ligand_sdf_file_name_list,
        target_center=tuple(args.center),
        option_yaml_file_name=args.configurations
    )

    docking_runner.run_unidock_protocol()

if __name__ == '__main__':
    main()
