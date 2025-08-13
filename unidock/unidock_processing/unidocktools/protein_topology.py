import msys

from rdkit.Chem import SplitMolByPDBResidues, GetMolFrags, FragmentOnBonds
from rdkit.Chem.PropertyMol import PropertyMol

from unidock_processing.utils.molecule_processing import (
    get_mol_with_indices,
    get_mol_without_indices,
)
from unidock_processing.atom_types.vina_atom_type import AtomType
from unidock_processing.unidocktools.supported_protein_residue_name import (
    PROTEIN_RESIUDE_NAME_LIST,
)


def is_peptide_bond(bond):
    """Checks if a bond is a peptide bond based on the residue_id and chain_id
    of the atoms on each part of the bond. Also works for disulfide bridges or any bond
    that links two residues in biopolymers.

    Parameters
    ----------
    bond: rdkit.Chem.rdchem.Bond
        The bond to check
    """

    begin_atom = bond.GetBeginAtom()
    end_atom = bond.GetEndAtom()

    begin_residue_idx = begin_atom.GetIntProp("internal_residue_idx")
    end_residue_idx = end_atom.GetIntProp("internal_residue_idx")

    begin_chain_idx = begin_atom.GetProp("chain_idx")
    end_chain_idx = end_atom.GetProp("chain_idx")

    if begin_residue_idx == end_residue_idx and begin_chain_idx == end_chain_idx:
        return False
    else:
        return True


def split_mol_by_residues(protein_mol):
    """Splits a protein_mol in multiple fragments based on residues

    Parameters
    ----------
    protein_mol: rdkit.Chem.Mol
        The protein molecule to fragment

    Returns
    -------
    residue_mol_list : list
        A list of :class:`rdkit.Chem.Mol` containing sorted residues of protein molecule
    """

    protein_residue_mol_list = []
    for residue_type_fragments in SplitMolByPDBResidues(protein_mol).values():
        for fragment in GetMolFrags(
            residue_type_fragments, asMols=True, sanitizeFrags=False
        ):
            # split on peptide bonds
            peptide_bond_idx_list = []
            for bond in fragment.GetBonds():
                if is_peptide_bond(bond):
                    peptide_bond_idx_list.append(bond.GetIdx())

            if len(peptide_bond_idx_list) > 0:
                splitted_mol = FragmentOnBonds(
                    fragment, peptide_bond_idx_list, addDummies=False
                )
                splitted_mol_list = GetMolFrags(
                    splitted_mol, asMols=True, sanitizeFrags=False
                )
                protein_residue_mol_list.extend(splitted_mol_list)
            else:
                protein_residue_mol_list.append(fragment)

    protein_residue_mol_dict = {}
    for protein_residue_mol in protein_residue_mol_list:
        atom = protein_residue_mol.GetAtomWithIdx(0)
        if atom.GetSymbol() == "H":
            continue

        protein_residue_mol_dict[atom.GetIntProp("internal_residue_idx")] = (
            protein_residue_mol
        )

    return [x[1] for x in sorted(protein_residue_mol_dict.items(), key=lambda x: x[0])]


def prepare_receptor_residue_mol_list(receptor_msys_system):
    num_receptor_atoms = receptor_msys_system.natoms
    receptor_nb_table = receptor_msys_system.getTable("nonbonded")

    if num_receptor_atoms != receptor_nb_table.nterms:
        raise ValueError("Problematic receptor preparation!!")

    receptor_atom_idx_list = [None] * num_receptor_atoms
    receptor_atom_name_list = [None] * num_receptor_atoms
    receptor_atom_charge_list = [None] * num_receptor_atoms
    receptor_resid_list = [None] * num_receptor_atoms
    receptor_resname_list = [None] * num_receptor_atoms
    receptor_chain_idx_list = [None] * num_receptor_atoms
    receptor_ff_atom_type_list = [None] * num_receptor_atoms
    receptor_internal_atom_idx_list = [None] * num_receptor_atoms
    receptor_internal_residue_idx_list = [None] * num_receptor_atoms

    for atom_idx in range(num_receptor_atoms):
        atom = receptor_msys_system.atom(atom_idx)
        receptor_atom_idx_list[atom_idx] = atom_idx + 1
        receptor_atom_name_list[atom_idx] = atom.name
        receptor_atom_charge_list[atom_idx] = atom.charge
        receptor_resid_list[atom_idx] = atom.residue.resid
        receptor_resname_list[atom_idx] = atom.residue.name
        receptor_chain_idx_list[atom_idx] = atom.residue.chain.name
        receptor_internal_atom_idx_list[atom_idx] = atom_idx
        receptor_internal_residue_idx_list[atom_idx] = atom.residue.id

        nb_term = receptor_nb_table.term(atom_idx)
        atom_type = nb_term["type"]
        receptor_ff_atom_type_list[atom_idx] = atom_type

    receptor_mol = msys.ConvertToRdkit(receptor_msys_system)

    receptor_positions = receptor_mol.GetConformer().GetPositions()
    num_receptor_mol_atoms = receptor_mol.GetNumAtoms()

    if num_receptor_atoms != num_receptor_mol_atoms:
        raise ValueError("Problematic msys receptor system conversion to rdkit mol!!")

    for atom_idx in range(num_receptor_atoms):
        atom = receptor_mol.GetAtomWithIdx(atom_idx)
        atom_positions = receptor_positions[atom_idx, :]

        atom.SetIntProp("atom_idx", int(receptor_atom_idx_list[atom_idx]))
        atom.SetProp("atom_name", receptor_atom_name_list[atom_idx])
        atom.SetDoubleProp("atom_charge", receptor_atom_charge_list[atom_idx])
        atom.SetProp("ff_atom_type", receptor_ff_atom_type_list[atom_idx])
        atom.SetIntProp("residue_idx", int(receptor_resid_list[atom_idx]))
        atom.SetProp("residue_name", receptor_resname_list[atom_idx])
        atom.SetProp("chain_idx", receptor_chain_idx_list[atom_idx])
        atom.SetIntProp("internal_atom_idx", receptor_internal_atom_idx_list[atom_idx])
        atom.SetIntProp(
            "internal_residue_idx", receptor_internal_residue_idx_list[atom_idx]
        )
        atom.SetDoubleProp("x", float(atom_positions[0]))
        atom.SetDoubleProp("y", float(atom_positions[1]))
        atom.SetDoubleProp("z", float(atom_positions[2]))

    non_protein_atom_idx_list = []
    for atom_idx in range(num_receptor_atoms):
        atom = receptor_mol.GetAtomWithIdx(atom_idx)
        receptor_resname = atom.GetProp("residue_name")
        if receptor_resname not in PROTEIN_RESIUDE_NAME_LIST:
            non_protein_atom_idx_list.append(atom_idx)

    protein_mol = get_mol_without_indices(
        receptor_mol,
        remove_indices=non_protein_atom_idx_list,
        keep_properties=[
            "atom_idx",
            "atom_name",
            "atom_charge",
            "ff_atom_type",
            "residue_idx",
            "residue_name",
            "chain_idx",
            "internal_atom_idx",
            "internal_residue_idx",
            "x",
            "y",
            "z",
        ],
    )

    cofactor_mol = get_mol_with_indices(
        receptor_mol,
        selected_indices=non_protein_atom_idx_list,
        keep_properties=[
            "atom_idx",
            "atom_name",
            "atom_charge",
            "ff_atom_type",
            "residue_idx",
            "residue_name",
            "chain_idx",
            "internal_atom_idx",
            "internal_residue_idx",
            "x",
            "y",
            "z",
        ],
    )

    atom_typer = AtomType()
    atom_typer.assign_atom_types(protein_mol)

    protein_residue_mol_list = split_mol_by_residues(protein_mol)

    protein_property_mol = PropertyMol(protein_mol)
    protein_residue_property_mol_list = [
        PropertyMol(protein_residue_mol)
        for protein_residue_mol in protein_residue_mol_list
    ]

    cofactor_residue_group_dict = {}
    num_cofactor_atoms = cofactor_mol.GetNumAtoms()
    for atom_idx in range(num_cofactor_atoms):
        atom = cofactor_mol.GetAtomWithIdx(atom_idx)
        internal_residue_idx = atom.GetIntProp("internal_residue_idx")
        if internal_residue_idx not in cofactor_residue_group_dict:
            cofactor_residue_group_dict[internal_residue_idx] = [atom_idx]
        else:
            cofactor_residue_group_dict[internal_residue_idx].append(atom_idx)

    cofactor_internal_residue_idx_list = list(cofactor_residue_group_dict.keys())
    num_cofactor_residues = len(cofactor_internal_residue_idx_list)

    cofactor_residue_property_mol_list = [None] * num_cofactor_residues
    for cofactor_idx in range(num_cofactor_residues):
        cofactor_internal_residue_idx = cofactor_internal_residue_idx_list[cofactor_idx]
        cofactor_atom_idx_list = cofactor_residue_group_dict[
            cofactor_internal_residue_idx
        ]
        cofactor_residue_mol = get_mol_with_indices(
            cofactor_mol,
            selected_indices=cofactor_atom_idx_list,
            keep_properties=[
                "atom_idx",
                "atom_name",
                "atom_charge",
                "ff_atom_type",
                "residue_idx",
                "residue_name",
                "chain_idx",
                "internal_atom_idx",
                "internal_residue_idx",
                "x",
                "y",
                "z",
            ],
        )

        atom_typer.assign_atom_types(cofactor_residue_mol)
        cofactor_residue_property_mol_list[cofactor_idx] = PropertyMol(
            cofactor_residue_mol
        )

    return (
        protein_property_mol,
        protein_residue_property_mol_list,
        cofactor_residue_property_mol_list,
    )
