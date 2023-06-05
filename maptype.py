from typing import Dict, List, Tuple, Callable, TypeVar, Generic
from enum import Enum
from maptools.core import LogicalTile, PhysicalTile
CIRTile = Tuple[int, int]
CIR2PhyIdxMap = Dict[CIRTile, int]
Logical2PhysicalMap = Dict[LogicalTile, PhysicalTile]

ArbitaryEdge = Tuple[PhysicalTile, PhysicalTile]
MeshEdge = Tuple[PhysicalTile, PhysicalTile]

class DLEMethod(Enum):
    REVERSE_S = 0

class DREMethod(Enum):
    DYXY = 0
    RPM = 1
    OCR = 2
