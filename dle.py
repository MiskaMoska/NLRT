from typing import Callable, List
from maptools.core import CTG, PhysicalTile
from acg import ACG
from abc import ABCMeta, abstractmethod
from random import shuffle
from maptype import CIR2PhyIdxMap, DLEMethod

class BaseDLE(Callable, metaclass=ABCMeta):
    '''
    Base class for Deterministic Layout Engine
    '''
    def __init__(self, ctg: CTG, acg: ACG) -> None:
        self.noc_w = acg.w
        self.noc_h = acg.h
        self.clusters = list(ctg.clusters)

    def __call__(self) -> CIR2PhyIdxMap:
        return self.map_tiles()

    @abstractmethod
    def generate_path(self) -> List[int]: ...
        
    def map_tiles(self) -> CIR2PhyIdxMap:
        map_dict = {}
        path = self.generate_path()
        for cidx, (base, cluster) in enumerate(self.clusters):
            id_list = list(range(len(cluster)))
            shuffle(id_list) # random mapping
            for i, tidx in enumerate(id_list):
                map_dict[(cidx, tidx)] = path[base + i]

        print(map_dict)
        return map_dict
            

class ReversesDLE(BaseDLE):

    def generate_path(self) -> List[int]:
        path = []
        for i in range(self.noc_h):
            for j in range(self.noc_w):
                idx = ((i+1) * self.noc_w-j-1 
                    if i % 2 else i*self.noc_w+j)
                path.append(idx)

        return path


__DLE_ACCESS_TABLE__ = {
    DLEMethod.REVERSE_S         :ReversesDLE
}