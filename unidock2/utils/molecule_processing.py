from rdkit import Chem


def _atom_copy_data(atom, atom_property_names):
    source_properties = atom.GetPropsAsDict()
    copied_properties = {
        property_name: source_properties[property_name]
        for property_name in atom_property_names
        if property_name in source_properties
    }

    atom_symbol = atom.GetSymbol()
    if atom_symbol.startswith("*"):
        normalized_symbol = "*"
        copied_properties["molAtomMapNumber"] = atom.GetAtomMapNum()
    elif atom_symbol.startswith("R"):
        normalized_symbol = "*"
        atom_map_number = int(atom_symbol[1:]) if len(atom_symbol) > 1 else atom.GetAtomMapNum()
        copied_properties["molAtomMapNumber"] = atom_map_number
        copied_properties["dummyLabel"] = f"R{atom_map_number}"
        copied_properties["_MolFileRLabel"] = str(atom_map_number)
    else:
        normalized_symbol = atom_symbol

    return (
        normalized_symbol,
        atom.GetChiralTag(),
        atom.GetFormalCharge(),
        atom.GetNumExplicitHs(),
        copied_properties,
    )


def _set_atom_properties(atom, properties):
    for property_name, property_value in properties.items():
        if isinstance(property_value, str):
            atom.SetProp(property_name, property_value)
        elif isinstance(property_value, int):
            atom.SetIntProp(property_name, property_value)
        elif isinstance(property_value, float):
            atom.SetDoubleProp(property_name, property_value)


def _copy_submolecule(
    mol_input,
    kept_indices,
    *,
    atom_property_names=(),
    molecule_property_names=(),
):
    """Copy selected atoms in source order using the legacy topology rules.

    Conformers and bond properties are intentionally not copied. Callers that
    need coordinates currently rebuild them from atom properties or construct a
    new conformer explicitly.
    """

    kept_index_set = set(kept_indices)
    molecule_properties = {property_name: mol_input.GetProp(property_name) for property_name in molecule_property_names}
    atom_data_list = [_atom_copy_data(atom, atom_property_names) for atom in mol_input.GetAtoms()]

    editable_molecule = Chem.RWMol(Chem.Mol())
    old_to_new_index = {}
    for old_atom_index, atom_data in enumerate(atom_data_list):
        if old_atom_index not in kept_index_set:
            continue

        copied_atom = Chem.Atom(atom_data[0])
        copied_atom.SetChiralTag(atom_data[1])
        copied_atom.SetFormalCharge(atom_data[2])
        copied_atom.SetNumExplicitHs(atom_data[3])
        _set_atom_properties(copied_atom, atom_data[4])
        old_to_new_index[old_atom_index] = editable_molecule.AddAtom(copied_atom)

    for source_bond in mol_input.GetBonds():
        begin_atom_index = source_bond.GetBeginAtomIdx()
        end_atom_index = source_bond.GetEndAtomIdx()
        begin_is_kept = begin_atom_index in kept_index_set
        end_is_kept = end_atom_index in kept_index_set

        if begin_is_kept and end_is_kept:
            editable_molecule.AddBond(
                old_to_new_index[begin_atom_index],
                old_to_new_index[end_atom_index],
                source_bond.GetBondType(),
            )
        elif begin_is_kept != end_is_kept:
            kept_atom_index = begin_atom_index if begin_is_kept else end_atom_index
            if atom_data_list[kept_atom_index][0] in ("N", "P"):
                kept_atom = editable_molecule.GetAtomWithIdx(old_to_new_index[kept_atom_index])
                kept_atom.SetNumExplicitHs(kept_atom.GetNumExplicitHs() + 1)

    copied_molecule = Chem.Mol(editable_molecule)
    for property_name, property_value in molecule_properties.items():
        copied_molecule.SetProp(property_name, property_value)

    Chem.GetSymmSSSR(copied_molecule)
    copied_molecule.UpdatePropertyCache(strict=False)
    return copied_molecule


def get_mol_without_indices(
    mol_input,
    remove_indices=(),
    keep_properties=(),
    keep_mol_properties=(),
):
    """Copy a molecule while excluding the requested source atom indices."""

    removed_index_set = set(remove_indices)
    kept_indices = (atom_index for atom_index in range(mol_input.GetNumAtoms()) if atom_index not in removed_index_set)
    return _copy_submolecule(
        mol_input,
        kept_indices,
        atom_property_names=keep_properties,
        molecule_property_names=keep_mol_properties,
    )


def get_mol_with_indices(
    mol_input,
    selected_indices=(),
    keep_properties=(),
    keep_mol_properties=(),
):
    """Copy only the requested source atom indices from a molecule."""

    return _copy_submolecule(
        mol_input,
        selected_indices,
        atom_property_names=keep_properties,
        molecule_property_names=keep_mol_properties,
    )
