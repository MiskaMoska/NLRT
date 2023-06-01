from typing import List
from maptools.core import PhysicalTile

class ACG(object):
    '''
    Architecture Characterization Graph 
    '''

    def __init__(self, w: int, h: int) -> None:
        self.w = w
        self.h = h

    @property
    def nodes(self) -> List[PhysicalTile]:
        return [(i, j) for i in range(self.w) for j in range(self.h)]
