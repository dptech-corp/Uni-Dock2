from abc import ABC, ABCMeta, abstractmethod
from rdkit import Chem


class MolGraphMeta(ABCMeta):
    def __init__(cls, name, bases, attrs):
        if not hasattr(cls, '_registry'):
            cls._registry = {}
        else:
            key = attrs.get('name', name).lower()
            cls._registry[key] = cls
        super().__init__(name, bases, attrs)


class BaseMolGraph(ABC, metaclass=MolGraphMeta):
    """Abstract base class for building molecule graph."""

    @abstractmethod
    def _preprocess_mol(self):
        raise NotImplementedError("Preprocessing mol method must be implemented.")

    @abstractmethod
    def _get_rotatable_bond_info(self) -> list[tuple[int,...]]:
        raise NotImplementedError("Rotatable bond identification method must be implemented.")

    @abstractmethod
    def _freeze_bond(self, rotatable_bond_info_list:list[tuple[int,...]]) -> list[Chem.Mol]:
        raise NotImplementedError("Bond freezing method must be implemented.")

    @abstractmethod
    def _get_root_atom_ids(self, splitted_mol_list:list[Chem.Mol],
                            rotatable_bond_info_list:list[tuple[int,...]]) -> list[int]:
        raise NotImplementedError("Root atom ID extraction method must be implemented.")

    @abstractmethod
    def build_graph(self):
        raise NotImplementedError("Rotatable bond identification method must be implemented.")

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> 'BaseMolGraph':
        try:
            SubCls = cls._registry[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown processor: {name!r}")
        return SubCls(*args, **kwargs)
