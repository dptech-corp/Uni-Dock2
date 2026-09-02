import os
from shutil import copyfile
from shutil import which

import msys
from openmm.app import Modeller, PDBFile
from pdbfixer import PDBFixer

from unidock2.utils.external_command import run_external_command


def parameterize_receptor(
    receptor_file_name,
    prepared_hydrogen,
    working_dir_name,
):
    """Create a parameterized receptor DMS using the existing preparation path."""
    working_dir_name = os.path.abspath(working_dir_name)
    receptor_structure_dms_file_name = os.path.join(
        working_dir_name, "receptor_structure.dms"
    )
    receptor_parameterized_dms_file_name = os.path.join(
        working_dir_name, "receptor_parameterized.dms"
    )
    fepfixer_executable = which("fepfixer")
    utop_executable = which("utop")
    if fepfixer_executable is not None and utop_executable is not None:
        fepfixer_command = [
            fepfixer_executable,
            "-i",
            os.path.abspath(receptor_file_name),
            "-o",
            os.path.basename(receptor_structure_dms_file_name),
        ]
        if prepared_hydrogen:
            fepfixer_command.append("--custom-protonation-states")

        run_external_command(
            fepfixer_command,
            cwd=working_dir_name,
            log_file_name="fepfixer.log",
            expected_output_file_names=[receptor_structure_dms_file_name],
        )
        run_external_command(
            [
                utop_executable,
                "prm",
                "-i",
                os.path.basename(receptor_structure_dms_file_name),
                "-o",
                os.path.basename(receptor_parameterized_dms_file_name),
            ],
            cwd=working_dir_name,
            log_file_name="utop.log",
            expected_output_file_names=[receptor_parameterized_dms_file_name],
        )
        return

    receptor_topology_preparation = ReceptorTopologyPreparation(
        receptor_file_name, working_dir_name
    )
    receptor_topology_preparation.run_preparation()


class ReceptorTopologyPreparation(object):
    def __init__(self, receptor_pdb_file_name, working_dir_name="."):
        self.receptor_pdb_file_name = receptor_pdb_file_name
        self.working_dir_name = os.path.abspath(working_dir_name)
        self.receptor_cleaned_pdb_file_name = os.path.join(
            self.working_dir_name, "receptor_cleaned.pdb"
        )
        self.receptor_fixed_pdb_file_name = os.path.join(
            self.working_dir_name, "receptor_fixed.pdb"
        )
        self.receptor_final_pdb_file_name = os.path.join(
            self.working_dir_name, "receptor_final.pdb"
        )
        self.receptor_prmtop_file_name = os.path.join(
            self.working_dir_name, "receptor.prmtop"
        )
        self.receptor_inpcrd_file_name = os.path.join(
            self.working_dir_name, "receptor.inpcrd"
        )
        self.receptor_dms_file_name = os.path.join(
            self.working_dir_name, "receptor_parameterized.dms"
        )

    def _run_tleap(self):
        tleap_source_file_name = os.path.join(
            os.path.dirname(__file__), "data", "tleap_receptor_template.in"
        )
        tleap_destination_file_name = os.path.join(self.working_dir_name, "tleap.in")
        copyfile(tleap_source_file_name, tleap_destination_file_name)

        run_external_command(
            ["tleap", "-f", "tleap.in"],
            cwd=self.working_dir_name,
            log_file_name="tleap.log",
            append_log=True,
            expected_output_file_names=[
                self.receptor_prmtop_file_name,
                self.receptor_inpcrd_file_name,
            ],
        )

    def run_preparation(self):
        receptor_pdb = PDBFile(self.receptor_pdb_file_name)
        modeller = Modeller(receptor_pdb.topology, receptor_pdb.positions)
        modeller.delete(
            [
                atom
                for atom in modeller.topology.atoms()
                if atom.name == "OXT" or atom.name.startswith("H")
            ]
        )

        with open(self.receptor_cleaned_pdb_file_name, "w") as receptor_cleaned_pdb_file:
            PDBFile.writeFile(
                modeller.topology,
                modeller.positions,
                receptor_cleaned_pdb_file,
                keepIds=True,
            )

        fixer = PDBFixer(filename=self.receptor_cleaned_pdb_file_name)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(True)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        with open(self.receptor_fixed_pdb_file_name, "w") as receptor_fixed_pdb_file:
            PDBFile.writeFile(fixer.topology, fixer.positions, receptor_fixed_pdb_file)

        for residue in fixer.topology.residues():
            if residue.name == "CYS":
                residue.name = "CYX"

        with open(self.receptor_final_pdb_file_name, "w") as receptor_final_pdb_file:
            PDBFile.writeFile(fixer.topology, fixer.positions, receptor_final_pdb_file)

        self._run_tleap()

        receptor_system = msys.LoadPrmTop(self.receptor_prmtop_file_name)
        msys.ReadCrdCoordinates(receptor_system, self.receptor_inpcrd_file_name)
        msys.AssignBondOrderAndFormalCharge(receptor_system)
        receptor_system.save(self.receptor_dms_file_name)
