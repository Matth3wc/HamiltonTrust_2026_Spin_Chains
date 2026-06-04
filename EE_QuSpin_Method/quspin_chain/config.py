from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ModelConfig:
    """Model specification for an XXZ chain.

    Attributes:
        L: chain length
        Jxy: nearest-neighbour XY coupling
        Jz: nearest-neighbour Z coupling
        Jxy2: next-nearest-neighbour XY coupling (NNN)
        Jz2: next-nearest-neighbour Z coupling (NNN)
        pbc: periodic boundary conditions flag
        basis_kwargs: forwarded to basis constructor
    """
    L: int
    Jxy: float = 1.0
    Jz: float = 1.0
    Jxy2: float = 0.0
    Jz2: float = 0.0
    pbc: bool = True
    basis_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepConfig:
    param_name: str
    param_values: List[float]
