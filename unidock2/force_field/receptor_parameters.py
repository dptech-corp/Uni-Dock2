"""Receptor force-field parameter handling."""

import math
import warnings

from unidock2.force_field.gaff2_mapping import FF_ATOM_TYPE_DICT


FALLBACK_RECEPTOR_FF_ATOM_TYPE = "c"
FALLBACK_RECEPTOR_CHARGE = 0.0


class MissingNonbondedTermsWarning(UserWarning):
    """Warning emitted when receptor force-field placeholders are used."""


def _safe_fallback_charge(value):
    try:
        charge = float(value)
    except (TypeError, ValueError, OverflowError):
        return FALLBACK_RECEPTOR_CHARGE
    return charge if math.isfinite(charge) else FALLBACK_RECEPTOR_CHARGE


def _read_receptor_force_field_data(receptor_msys_system):
    """Read nonbonded types, or provide Vina-compatible placeholders when absent."""
    num_receptor_atoms = receptor_msys_system.natoms
    receptor_nb_table = receptor_msys_system.getTable("nonbonded")
    num_nonbonded_terms = 0 if receptor_nb_table is None else receptor_nb_table.nterms

    if num_nonbonded_terms not in (0, num_receptor_atoms):
        raise ValueError(
            "Problematic receptor preparation: "
            f"nonbonded term count ({num_nonbonded_terms}) does not match "
            f"atom count ({num_receptor_atoms})."
        )

    if num_nonbonded_terms == num_receptor_atoms and num_receptor_atoms > 0:
        ff_atom_types = [
            receptor_nb_table.term(atom_idx)["type"]
            for atom_idx in range(num_receptor_atoms)
        ]
        charges = [
            receptor_msys_system.atom(atom_idx).charge
            for atom_idx in range(num_receptor_atoms)
        ]
        return ff_atom_types, charges

    warnings.warn(
        "Receptor has no nonbonded terms; using fallback FF atom type "
        f"'{FALLBACK_RECEPTOR_FF_ATOM_TYPE}' and atom charges, with "
        f"{FALLBACK_RECEPTOR_CHARGE} for invalid charges.",
        MissingNonbondedTermsWarning,
        stacklevel=2,
    )
    ff_atom_types = [FALLBACK_RECEPTOR_FF_ATOM_TYPE] * num_receptor_atoms
    charges = [
        _safe_fallback_charge(receptor_msys_system.atom(atom_idx).charge)
        for atom_idx in range(num_receptor_atoms)
    ]
    return ff_atom_types, charges


def assign_receptor_force_field_properties(atom, atom_type, charge):
    """Attach the legacy receptor force-field fields to an RDKit atom."""
    atom.SetDoubleProp("atom_charge", charge)
    atom.SetProp("ff_atom_type", atom_type)


def encode_receptor_force_field(atom):
    """Return the force-field slots required by the native molecule protocol."""
    return (
        FF_ATOM_TYPE_DICT[atom.GetProp("ff_atom_type")],
        atom.GetDoubleProp("atom_charge"),
    )
