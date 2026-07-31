import os
from shutil import copyfile
import MDAnalysis as mda
import msys

from pdbfixer import PDBFixer
from openmm.app import PDBFile


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

    def run_preparation(self):
        receptor_ag = mda.Universe(self.receptor_pdb_file_name).atoms
        cleaned_ag = receptor_ag.select_atoms("not name OXT and not name H*")
        cleaned_ag.write(self.receptor_cleaned_pdb_file_name)

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

        fixed_ag = mda.Universe(self.receptor_fixed_pdb_file_name).atoms
        for residue in fixed_ag.residues:
            if residue.resname == "CYS":
                residue.resname = "CYX"

        fixed_ag.write(self.receptor_final_pdb_file_name)

        tleap_source_file_name = os.path.join(
            os.path.dirname(__file__), "data", "tleap_receptor_template.in"
        )
        tleap_destination_file_name = os.path.join(self.working_dir_name, "tleap.in")
        copyfile(tleap_source_file_name, tleap_destination_file_name)

        tleap_command = (
            f"cd {self.working_dir_name}; "
            "tleap -f tleap.in >> tleap.log; cd - >> tleap.log"
        )
        os.system(tleap_command)

        receptor_system = msys.LoadPrmTop(self.receptor_prmtop_file_name)
        msys.ReadCrdCoordinates(receptor_system, self.receptor_inpcrd_file_name)
        msys.AssignBondOrderAndFormalCharge(receptor_system)
        receptor_system.save(self.receptor_dms_file_name)
