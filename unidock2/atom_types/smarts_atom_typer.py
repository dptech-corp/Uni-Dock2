"""Reusable SMARTS-based atom typing."""

from collections.abc import Iterable
from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class SmartsRule:
    """Assign an atom type to the first atom matched by a SMARTS pattern."""

    smarts: str
    atom_type: str


class SmartsAtomTyper:
    """Apply ordered SMARTS rules, with later rules overriding earlier rules."""

    def __init__(self, rules: Iterable[SmartsRule], property_name: str):
        self.rules = tuple(rules)
        self.property_name = property_name
        self._compiled_rules = tuple(self._compile_rule(rule) for rule in self.rules)

    @staticmethod
    def _compile_rule(rule: SmartsRule):
        pattern = Chem.MolFromSmarts(rule.smarts)
        if pattern is None:
            raise ValueError(f"Invalid SMARTS for atom type {rule.atom_type!r}: {rule.smarts!r}")
        return pattern, rule.atom_type

    def assign_atom_types(self, mol: Chem.Mol) -> None:
        """Assign atom types in place on an RDKit molecule."""
        assigned_atom_indices = set()

        for pattern, atom_type in self._compiled_rules:
            for pattern_match in mol.GetSubstructMatches(pattern, maxMatches=1_000_000):
                atom_idx = pattern_match[0]
                mol.GetAtomWithIdx(atom_idx).SetProp(self.property_name, atom_type)
                assigned_atom_indices.add(atom_idx)

        unassigned_atom_indices = [
            atom_idx for atom_idx in range(mol.GetNumAtoms()) if atom_idx not in assigned_atom_indices
        ]
        if unassigned_atom_indices:
            raise ValueError(f"SMARTS atom typing did not assign atom indices: {unassigned_atom_indices}")
