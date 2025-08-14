from abc import ABC, ABCMeta, abstractmethod
from rdkit import Chem


class RotatableBondMeta(ABCMeta):
    def __init__(cls, name, bases, attrs):
        if not hasattr(cls, '_registry'):
            cls._registry = {}
        else:
            key = attrs.get('name', name).lower()
            cls._registry[key] = cls
        super().__init__(name, bases, attrs)


class BaseRotatableBond(ABC, metaclass=RotatableBondMeta):
    """Abstract base class for rotatable bond identification."""

    @abstractmethod
    def identify_rotatable_bonds(self, mol:Chem.Mol):
        """
        Identify rotatable bonds in a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            list: List of tuples containing atom indices of rotatable bonds
        """
        raise NotImplementedError("Rotatable bond identification method must be implemented.")

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> 'BaseRotatableBond':
        try:
            SubCls = cls._registry[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown processor: {name!r}")
        return SubCls(*args, **kwargs)
