import sys
from importlib import import_module
from importlib.metadata import version
import argparse

logo_description = r"""

    ██╗   ██╗██████╗ ██████╗ 
    ██║   ██║██╔══██╗╚════██╗
    ██║   ██║██║  ██║ █████╔╝
    ██║   ██║██║  ██║██╔═══╝ 
    ╚██████╔╝██████╔╝███████╗
     ╚═════╝ ╚═════╝ ╚══════╝

    DP Technology Docking Toolkit

"""

available_commands = [
    ('docking', 'unidock_processing.cli.docking'),
    ('protein_prep','unidock_processing.cli.protein_prep')
]

class CLIDriver(object):
    def __init__(self):
        self._command_table = None
        self._argument_table = None

    def run(self,
             prog='unidock2',
             commands=available_commands,
             args=None):

        print(logo_description)
        parser = argparse.ArgumentParser(prog=prog,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         add_help=False)

        parser.add_argument(
        '-h',
        '--help',
        action='store_true',
        help='Show global help message')

        parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f"%(prog)s {version('unidock_processing')}",
        help='Show program version')

        subparsers = parser.add_subparsers(title='Sub-commands',
                                           dest='command',
                                           help='Uni-Dock2-related applications',
                                           required=False)

        help_parser = subparsers.add_parser('help',
                                            description='Show help for sub-commands',
                                            help='Detailed help for sub-commands')

        help_parser.add_argument('subcommand',
                                 nargs='?',
                                 metavar='SUBCOMMAND',
                                 help='Name of the subcommand to show help for')

        name_module_dict = {}
        subparser_dict = {}
        for command, module_name in commands:
            module = import_module(module_name).CLICommand
            docstring = module.__doc__ or ''
            cmd_parser = subparsers.add_parser(command,
                                              description=docstring,
                                              formatter_class=argparse.RawDescriptionHelpFormatter)

            module.add_arguments(cmd_parser)
            name_module_dict[command] = module
            subparser_dict[command] = cmd_parser

        args = parser.parse_args()

        if args.help:
            parser.print_help()
            sys.exit(0)

        if args.command == 'help':
            if args.subcommand:
                if args.subcommand in subparser_dict:
                    subparser_dict[args.subcommand].print_help()
                else:
                    raise ValueError(f'Unknown sub-command: {args.subcommand}')

            else:
                parser.print_help()

            sys.exit(0)

        if args.command is None:
            parser.print_help()
            sys.exit(0)

        name_module_dict[args.command].run(args)

def main():
    driver = CLIDriver()
    rc = driver.run()
    return rc

if __name__ == '__main__':
    main()
