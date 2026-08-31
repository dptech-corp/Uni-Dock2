import os
from copy import deepcopy
from pathlib import Path

from rdkit import Chem

from unidock2.force_field.gaff2_mapping import FF_ATOM_TYPE_DICT
from unidock2.utils.external_command import run_external_command


def convert_v3000_mol_to_v2000_sdf(v3000_mol, v2000_sdf_file_name):
    mol_props_dict = v3000_mol.GetPropsAsDict()
    v2000_str = Chem.MolToV2KMolBlock(v3000_mol)
    for prop_key, prop_value in mol_props_dict.items():
        v2000_str += f">  <{prop_key}>  (1) \n"
        v2000_str += f"{prop_value}\n\n"

    with open(v2000_sdf_file_name, "w") as f:
        f.write(v2000_str)


def _run_ambertools_for_gaff2(
    working_dir_name,
    ligand_sdf_file_name,
    ligand_mol2_file_name,
    ligand_frcmod_file_name,
    ligand_charge_method,
    formal_charge,
):
    working_dir = Path(working_dir_name)
    antechamber_frcmod_file_name = working_dir / "ANTECHAMBER.FRCMOD"

    try:
        run_external_command(
            [
                "antechamber",
                "-i",
                Path(ligand_sdf_file_name).name,
                "-fi",
                "sdf",
                "-o",
                Path(ligand_mol2_file_name).name,
                "-fo",
                "mol2",
                "-at",
                "gaff2",
                "-c",
                ligand_charge_method,
                "-nc",
                str(formal_charge),
                "-eq",
                "2",
                "-pf",
                "y",
            ],
            cwd=working_dir,
            log_file_name="ligand_temp_antechamber.log",
            append_log=True,
            expected_output_file_names=[ligand_mol2_file_name],
        )
        run_external_command(
            [
                "parmchk2",
                "-i",
                Path(ligand_mol2_file_name).name,
                "-f",
                "mol2",
                "-a",
                "Y",
                "-s",
                "2",
                "-o",
                Path(ligand_frcmod_file_name).name,
            ],
            cwd=working_dir,
            log_file_name="ligand_temp_parmchk2.log",
            expected_output_file_names=[ligand_frcmod_file_name],
        )
    finally:
        antechamber_frcmod_file_name.unlink(missing_ok=True)


# Preserve the legacy Antechamber preprocessing workaround: protonate selected
# anionic nitrogens in the parameterization copy before atom typing/charge assignment.
_ANIONIC_NITROGEN_SMARTS_LIST = [
    "[$(NS=O),$(NP=O);-1]",  # sulfonamide and phosphonamide
    "[n;H0;-1]",  # any aromatic anionic N; historically intended for tetrazoles
]


def _protonate_anionic_nitrogens(mol):
    for anionic_nitrogen_smarts in _ANIONIC_NITROGEN_SMARTS_LIST:
        anionic_nitrogen_pattern = Chem.MolFromSmarts(anionic_nitrogen_smarts)
        nitrogen_match_tuple_list = mol.GetSubstructMatches(anionic_nitrogen_pattern)

        for nitrogen_match_tuple in nitrogen_match_tuple_list:
            atom = mol.GetAtomWithIdx(nitrogen_match_tuple[0])
            num_implicit_Hs = atom.GetNumImplicitHs()
            num_explicit_Hs = atom.GetNumExplicitHs()
            total_current_num_Hs = num_implicit_Hs + num_explicit_Hs
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(total_current_num_Hs + 1)


def record_gaff2_atom_types_and_parameters(ligand_sdf_file_name, ligand_charge_method, working_dir_name):
    working_dir_name = os.path.abspath(working_dir_name)

    mol = Chem.SDMolSupplier(ligand_sdf_file_name, removeHs=False)[0]
    num_atoms = mol.GetNumAtoms()

    mol_copy = deepcopy(mol)
    _protonate_anionic_nitrogens(mol_copy)

    Chem.GetSymmSSSR(mol_copy)
    mol_copy.UpdatePropertyCache(strict=False)

    # AddHs for both sulfonamide tetrazole cases and covalent ligand dummy hydrogens.
    mol_copy_h = Chem.AddHs(mol_copy, addCoords=True)
    formal_charge = Chem.GetFormalCharge(mol_copy_h)

    temp_ligand_sdf_file_name = os.path.join(working_dir_name, "ligand_temp.sdf")
    temp_ligand_mol2_file_name = os.path.join(working_dir_name, "ligand_temp.mol2")
    temp_ligand_frcmod_file_name = os.path.join(working_dir_name, "ligand_temp.frcmod")

    convert_v3000_mol_to_v2000_sdf(mol_copy_h, temp_ligand_sdf_file_name)

    ## Execute ambertools
    _run_ambertools_for_gaff2(
        working_dir_name,
        temp_ligand_sdf_file_name,
        temp_ligand_mol2_file_name,
        temp_ligand_frcmod_file_name,
        ligand_charge_method,
        formal_charge,
    )

    ## Record atom types and parameters
    ## mol2 file parsing
    atom_type_list = [None] * num_atoms
    partial_charge_list = [None] * num_atoms
    atom_parameter_dict = {}
    torsion_parameter_dict = {}

    with open(temp_ligand_mol2_file_name) as mol2_file:
        line_list = mol2_file.readlines()

    for line_idx, line in enumerate(line_list):
        if line.startswith("@<TRIPOS>ATOM"):
            atom_header_line_idx = line_idx
        elif line.startswith("@<TRIPOS>BOND"):
            bond_header_line_idx = line_idx

    atom_idx = 0
    for line_idx in range(atom_header_line_idx + 1, bond_header_line_idx):
        line = line_list[line_idx]
        line_split_list = line.strip().split()
        atom_type = line_split_list[5]
        partial_charge = float(line_split_list[8])

        atom_type_list[atom_idx] = atom_type
        partial_charge_list[atom_idx] = partial_charge
        atom_idx += 1

        if atom_idx == num_atoms:
            break

    ## frcmod file parsing
    ## ambertools frcmod format refer to https://ambermd.org/FileFormats.php
    with open(temp_ligand_frcmod_file_name) as frcmod_file:
        line_list = frcmod_file.readlines()

    num_frcmod_lines = len(line_list)

    for line_idx, line in enumerate(line_list):
        if line.startswith("MASS"):
            atom_header_line_idx = line_idx
        elif line.startswith("BOND"):
            bond_header_line_idx = line_idx

    for line_idx in range(atom_header_line_idx + 1, bond_header_line_idx):
        line = line_list[line_idx]
        if len(line) < 4:
            continue

        line_split_list = line.strip().split()
        atom_type = line_split_list[0]
        mass = float(line_split_list[1])
        atom_parameter_dict[atom_type] = {}
        atom_parameter_dict[atom_type]["mass"] = mass

    for line_idx, line in enumerate(line_list):
        if line.startswith("NONBON"):
            nonbond_header_line_idx = line_idx

    for line_idx in range(nonbond_header_line_idx + 1, num_frcmod_lines):
        line = line_list[line_idx]
        if len(line) < 4:
            continue

        line_split_list = line.strip().split()
        atom_type = line_split_list[0]
        sigma = float(line_split_list[1])
        epsilon = float(line_split_list[2])
        atom_parameter_dict[atom_type]["sigma"] = sigma
        atom_parameter_dict[atom_type]["epsilon"] = epsilon

    for line_idx, line in enumerate(line_list):
        if line.startswith("DIHE"):
            torsion_header_line_idx = line_idx
        elif line.startswith("IMPROPER"):
            improper_header_line_idx = line_idx

    for line_idx in range(torsion_header_line_idx + 1, improper_header_line_idx):
        line = line_list[line_idx]
        if len(line) < 4:
            continue

        torsion_type_str = line[:11]
        torsion_type_split_list = torsion_type_str.split("-")
        torsion_type_i = torsion_type_split_list[0].strip()
        torsion_type_j = torsion_type_split_list[1].strip()
        torsion_type_k = torsion_type_split_list[2].strip()
        torsion_type_l = torsion_type_split_list[3].strip()
        torsion_type_tuple = (
            torsion_type_i,
            torsion_type_j,
            torsion_type_k,
            torsion_type_l,
        )

        torsion_parameter_str = line[14:54]
        torsion_parameter_split_list = torsion_parameter_str.strip().split()

        fps_dict = {}
        fps_dict["barrier_factor"] = int(torsion_parameter_split_list[0])
        fps_dict["barrier_height"] = float(torsion_parameter_split_list[1])
        fps_dict["periodicity"] = int(abs(float(torsion_parameter_split_list[3])))
        fps_dict["phase"] = float(torsion_parameter_split_list[2])

        if torsion_type_tuple in torsion_parameter_dict.keys():
            torsion_parameter_dict[torsion_type_tuple].append(fps_dict)
        else:
            torsion_parameter_dict[torsion_type_tuple] = [fps_dict]
    ##############################################################################

    return (
        atom_type_list,
        partial_charge_list,
        atom_parameter_dict,
        torsion_parameter_dict,
    )


def construct_gaff2(mol, working_dir_name):
    temp_ligand_sdf_file_name = os.path.join(working_dir_name, "ligand.sdf")
    with Chem.SDWriter(temp_ligand_sdf_file_name) as writer:
        writer.write(mol)

    (
        atom_type_list,
        partial_charge_list,
        atom_parameter_dict,
        torsion_parameter_nested_dict,
    ) = record_gaff2_atom_types_and_parameters(temp_ligand_sdf_file_name, "gas", working_dir_name)
    return atom_type_list, partial_charge_list, torsion_parameter_nested_dict


def get_ligand_force_field_data(mol, construct_ff, working_dir_name):
    if construct_ff:
        return construct_gaff2(mol, working_dir_name)

    num_atoms = mol.GetNumAtoms()
    atom_type_list = ["c"] * num_atoms
    partial_charge_list = [0.0] * num_atoms
    torsion_parameter_nested_dict = {}
    return atom_type_list, partial_charge_list, torsion_parameter_nested_dict


def encode_atom_force_field(atom_type, partial_charge):
    return FF_ATOM_TYPE_DICT[atom_type], partial_charge


def get_torsion_force_field_parameters(
    torsion_atom_idx_list,
    atom_type_list,
    torsion_parameter_nested_dict,
    construct_ff,
):
    if construct_ff:
        torsion_type = [atom_type_list[torsion_atom_idx_list[i]] for i in range(4)]
        if tuple(torsion_type) not in torsion_parameter_nested_dict:
            torsion_type = reversed(torsion_type)
        torsion_parameter_dict_list = torsion_parameter_nested_dict[tuple(torsion_type)]

        return [
            [
                torsion_parameter_dict["barrier_factor"],
                torsion_parameter_dict["barrier_height"],
                torsion_parameter_dict["periodicity"],
                torsion_parameter_dict["phase"],
            ]
            for torsion_parameter_dict in torsion_parameter_dict_list
        ]

    return []
