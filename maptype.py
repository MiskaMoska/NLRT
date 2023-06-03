from typing import Dict, List, Tuple, Callable, TypeVar, Generic
from maptools.core import LogicalTile, PhysicalTile
CIRTile = Tuple[int, int]
CIR2PhyIdxMap = Dict[CIRTile, int]
Logical2PhysicalMap = Dict[LogicalTile, PhysicalTile]

ArbitaryEdge = Tuple[PhysicalTile, PhysicalTile]
MeshEdge = Tuple[PhysicalTile, PhysicalTile]


