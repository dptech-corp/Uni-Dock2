from rdkit import Chem

ATOM_TYPE_DEFINITION_LIST = [
    {"smarts": "[#1]", "atype": "H"},
    {"smarts": "[#5]", "atype": "B"},
    {"smarts": "[#6]", "atype": "C_H"},
    {
        "smarts": "[#6;$([#6]~[#5,#7,#8,#9,#14,#15,#16,#17,#34,#35,#53])]",
        "atype": "C_P",
    },
    {"smarts": "[#7]", "atype": "N_P"},
    {"smarts": "[#7;!H0]", "atype": "N_D"},
    {
        "smarts": (
            "[#7;!$([#7X3v3][a]);!$([#7X3v3][#6X3v4]);"
            "!$([#7X3v3][NX2]=[*]);!$([#7+1])]"
        ),
        "atype": "N_A",
    },
    {
        "smarts": (
            "[#7;!$([#7X3v3][a]);!$([#7X3v3][#6X3v4]);"
            "!$([#7X3v3][NX2]=[*]);!$([#7+1]);!H0]"
        ),
        "atype": "N_DA",
    },
    {"smarts": "[#8]", "atype": "O_A"},
    {"smarts": "[O;!H0]", "atype": "O_DA"},
    {"smarts": "[#9]", "atype": "F_H"},
    {"smarts": "[#14]", "atype": "Si"},
    {"smarts": "[#15]", "atype": "P_P"},
    {"smarts": "[#16]", "atype": "S_P"},
    {"smarts": "[#17]", "atype": "Cl_H"},
    {"smarts": "[#35]", "atype": "Br_H"},
    {"smarts": "[#53]", "atype": "I_H"},
    {"smarts": "[#85]", "atype": "At"},
    {
        "smarts": "[!#1;!#5;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53;!#85]",
        "atype": "Met_D",
    },
]


class AtomType(object):
    def __init__(self):
        self.atom_type_definition_list = ATOM_TYPE_DEFINITION_LIST

    def assign_atom_types(self, mol):
        for atom_type_dict in self.atom_type_definition_list:
            smarts = atom_type_dict["smarts"]
            atom_type = atom_type_dict["atype"]

            pattern_mol = Chem.MolFromSmarts(smarts)
            pattern_matches = mol.GetSubstructMatches(pattern_mol, maxMatches=1000000)
            for pattern_match in pattern_matches:
                atom = mol.GetAtomWithIdx(pattern_match[0])
                atom.SetProp("vina_atom_type", atom_type)
