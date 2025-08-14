from abc import ABC, ABCMeta, abstractmethod


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
    def build_graph(self):
        """
        Identify rotatable bonds in a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            list: List of tuples containing atom indices of rotatable bonds
        """
        raise NotImplementedError("Rotatable bond identification method must be implemented.")

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> 'BaseMolGraph':
        try:
            SubCls = cls._registry[name.lower()]
        except KeyError:
            raise ValueError(f"Unknown processor: {name!r}")
        return SubCls(*args, **kwargs)
