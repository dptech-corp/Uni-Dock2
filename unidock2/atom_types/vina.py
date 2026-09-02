"""Vina atom-type rules and engine type identifiers."""

from unidock2.atom_types.smarts_atom_typer import SmartsAtomTyper, SmartsRule

VINA_ATOM_TYPE_PROPERTY = "vina_atom_type"

VINA_ATOM_TYPE_RULES = (
    SmartsRule("[#1]", "H"),
    SmartsRule("[#5]", "B"),
    SmartsRule("[#6]", "C_H"),
    SmartsRule(
        "[#6;$([#6]~[#5,#7,#8,#9,#14,#15,#16,#17,#34,#35,#53])]",
        "C_P",
    ),
    SmartsRule("[#7]", "N_P"),
    SmartsRule("[#7;!H0]", "N_D"),
    SmartsRule(
        "[#7;!$([#7X3v3][a]);!$([#7X3v3][#6X3v4]);!$([#7X3v3][NX2]=[*]);!$([#7+1])]",
        "N_A",
    ),
    SmartsRule(
        "[#7;!$([#7X3v3][a]);!$([#7X3v3][#6X3v4]);!$([#7X3v3][NX2]=[*]);!$([#7+1]);!H0]",
        "N_DA",
    ),
    SmartsRule("[#8]", "O_A"),
    SmartsRule("[O;!H0]", "O_DA"),
    SmartsRule("[#9]", "F_H"),
    SmartsRule("[#14]", "Si"),
    SmartsRule("[#15]", "P_P"),
    SmartsRule("[#16]", "S_P"),
    SmartsRule("[#17]", "Cl_H"),
    SmartsRule("[#35]", "Br_H"),
    SmartsRule("[#53]", "I_H"),
    SmartsRule("[#85]", "At"),
    SmartsRule(
        "[!#1;!#5;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53;!#85]",
        "Met_D",
    ),
)

# O_P and O_D remain part of the Python-engine protocol even though the
# current SMARTS rules do not assign them.
VINA_ATOM_TYPE_DICT = {
    "H": 0,
    "B": 1,
    "C_H": 2,
    "C_P": 3,
    "N_P": 4,
    "N_D": 5,
    "N_A": 6,
    "N_DA": 7,
    "O_P": 8,
    "O_D": 9,
    "O_A": 10,
    "O_DA": 11,
    "F_H": 12,
    "Si": 13,
    "P_P": 14,
    "S_P": 15,
    "Cl_H": 16,
    "Br_H": 17,
    "I_H": 18,
    "At": 19,
    "Met_D": 20,
}


class VinaAtomTyper(SmartsAtomTyper):
    """Assign the Vina atom types consumed by the engine."""

    def __init__(self):
        super().__init__(
            rules=VINA_ATOM_TYPE_RULES,
            property_name=VINA_ATOM_TYPE_PROPERTY,
        )
