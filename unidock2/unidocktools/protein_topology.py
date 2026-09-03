import msys

from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol

from unidock2.utils.molecule_processing import (
    get_mol_with_indices,
    get_mol_without_indices,
    set_atom_properties,
)
from unidock2.atom_types.vina import VinaAtomTyper
# Legacy import paths remain available while implementation stays in force_field.
from unidock2.force_field.receptor_parameters import (
    FALLBACK_RECEPTOR_CHARGE as FALLBACK_RECEPTOR_CHARGE,
    FALLBACK_RECEPTOR_FF_ATOM_TYPE as FALLBACK_RECEPTOR_FF_ATOM_TYPE,
    MissingNonbondedTermsWarning as MissingNonbondedTermsWarning,
    _read_receptor_force_field_data as _read_receptor_force_field_data,
    assign_receptor_force_field_properties,
)
from unidock2.unidocktools.supported_protein_residue_name import (
    PROTEIN_RESIUDE_NAME_LIST,
)

RECEPTOR_ATOM_PROPERTY_NAMES = (
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


def _build_residue_mol(atom_data_list, atom_idx_list, residue_bond_list):
    """Assemble one residue from atom data already snapshotted from the parent.

    Only the residue's own atoms and bonds are touched, so the cost is
    proportional to the residue rather than to the whole protein.
    """

    editable_mol = Chem.RWMol(Chem.Mol())
    parent_to_residue_atom_idx_dict = {}
    for parent_atom_idx in atom_idx_list:
        atom_symbol, chiral_tag, formal_charge, num_explicit_hs, properties = (
            atom_data_list[parent_atom_idx]
        )
        atom = Chem.Atom(atom_symbol)
        atom.SetChiralTag(chiral_tag)
        atom.SetFormalCharge(formal_charge)
        atom.SetNumExplicitHs(num_explicit_hs)
        set_atom_properties(atom, properties)
        parent_to_residue_atom_idx_dict[parent_atom_idx] = editable_mol.AddAtom(atom)

    for begin_atom_idx, end_atom_idx, bond_type in residue_bond_list:
        editable_mol.AddBond(
            parent_to_residue_atom_idx_dict[begin_atom_idx],
            parent_to_residue_atom_idx_dict[end_atom_idx],
            bond_type,
        )

    residue_mol = Chem.Mol(editable_mol)
    Chem.GetSymmSSSR(residue_mol)
    residue_mol.UpdatePropertyCache(strict=False)
    return residue_mol


def split_mol_by_residues(protein_mol):
    """Splits a protein_mol in multiple fragments based on residues

    Every atom already records the residue it belongs to, so the residues are
    read from ``internal_residue_idx`` in a single pass over the atoms and a
    single pass over the bonds. Recovering them from connectivity instead
    requires RDKit to materialize one fragment at a time out of the whole
    protein, which costs a full copy of ``protein_mol`` per residue.

    Parameters
    ----------
    protein_mol: rdkit.Chem.Mol
        The protein molecule to fragment

    Returns
    -------
    residue_mol_list : list
        A list of :class:`rdkit.Chem.Mol` containing sorted residues of protein molecule
    """

    num_protein_atoms = protein_mol.GetNumAtoms()
    atom_residue_idx_list = [None] * num_protein_atoms
    atom_data_list = [None] * num_protein_atoms
    residue_atom_idx_dict = {}

    for atom in protein_mol.GetAtoms():
        atom_idx = atom.GetIdx()
        internal_residue_idx = atom.GetIntProp("internal_residue_idx")
        atom_residue_idx_list[atom_idx] = internal_residue_idx
        residue_atom_idx_dict.setdefault(internal_residue_idx, []).append(atom_idx)
        atom_data_list[atom_idx] = (
            atom.GetSymbol(),
            atom.GetChiralTag(),
            atom.GetFormalCharge(),
            atom.GetNumExplicitHs(),
            atom.GetPropsAsDict(),
        )

    # Bonds that join two residues are dropped, matching a split on those bonds.
    residue_bond_dict = {
        internal_residue_idx: [] for internal_residue_idx in residue_atom_idx_dict
    }
    for bond in protein_mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        internal_residue_idx = atom_residue_idx_list[begin_atom_idx]
        if internal_residue_idx == atom_residue_idx_list[end_atom_idx]:
            residue_bond_dict[internal_residue_idx].append(
                (begin_atom_idx, end_atom_idx, bond.GetBondType())
            )

    protein_residue_mol_dict = {}
    for internal_residue_idx, atom_idx_list in residue_atom_idx_dict.items():
        if atom_data_list[atom_idx_list[0]][0] == "H":
            continue

        protein_residue_mol_dict[internal_residue_idx] = _build_residue_mol(
            atom_data_list,
            atom_idx_list,
            residue_bond_dict[internal_residue_idx],
        )

    return [x[1] for x in sorted(protein_residue_mol_dict.items(), key=lambda x: x[0])]


def prepare_receptor_residue_mol_list(receptor_msys_system):
    num_receptor_atoms = receptor_msys_system.natoms
    receptor_ff_atom_type_list, receptor_atom_charge_list = (
        _read_receptor_force_field_data(receptor_msys_system)
    )

    receptor_atom_idx_list = [None] * num_receptor_atoms
    receptor_atom_name_list = [None] * num_receptor_atoms
    receptor_resid_list = [None] * num_receptor_atoms
    receptor_resname_list = [None] * num_receptor_atoms
    receptor_chain_idx_list = [None] * num_receptor_atoms
    receptor_internal_atom_idx_list = [None] * num_receptor_atoms
    receptor_internal_residue_idx_list = [None] * num_receptor_atoms

    for atom_idx in range(num_receptor_atoms):
        atom = receptor_msys_system.atom(atom_idx)
        receptor_atom_idx_list[atom_idx] = atom_idx + 1
        receptor_atom_name_list[atom_idx] = atom.name
        receptor_resid_list[atom_idx] = atom.residue.resid
        receptor_resname_list[atom_idx] = atom.residue.name
        receptor_chain_idx_list[atom_idx] = atom.residue.chain.name
        receptor_internal_atom_idx_list[atom_idx] = atom_idx
        receptor_internal_residue_idx_list[atom_idx] = atom.residue.id

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
        assign_receptor_force_field_properties(
            atom,
            receptor_ff_atom_type_list[atom_idx],
            receptor_atom_charge_list[atom_idx],
        )
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
        keep_properties=RECEPTOR_ATOM_PROPERTY_NAMES,
    )

    cofactor_mol = get_mol_with_indices(
        receptor_mol,
        selected_indices=non_protein_atom_idx_list,
        keep_properties=RECEPTOR_ATOM_PROPERTY_NAMES,
    )

    atom_typer = VinaAtomTyper()
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
            keep_properties=RECEPTOR_ATOM_PROPERTY_NAMES,
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
