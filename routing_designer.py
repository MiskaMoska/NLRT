import random
from maptools.core import CTG, LogicalTile, PhysicalTile
from acg import ACG
import numpy as np
from typing import List, Dict, Tuple, Any
from maptype import CIRTile, CIR2PhyIdxMap, Logical2PhysicalMap
from layout_designer import LayoutResult
from encoding import RoutingPatternCode
from algorithm import RoutingSimulatedAnnealing
from routing_result import RoutingResult

class RoutingDesigner(object):

    def __init__(self, ctg: CTG, acg: ACG, layout: LayoutResult) -> None:
        '''
        Tile-NoC Layout Designer
        Determines the 1-to-1 mapping between logical tile and physical tiles

        Parameter
        ---------
        ctg: CTG
            Communication Trace Graph of the AI task.

        acg: ACG
            Architecture Characterization Graph of the NoC.

        layout: LayoutResult
            layout result from `LayoutDesigner`.
        '''
        self.noc_w = acg.w
        self.noc_h = acg.h
        self.layout = layout
        self.rpc = RoutingPatternCode(ctg, acg, layout)

    def obj_func(self, x: RoutingPatternCode) -> float:
        '''
        objective function for optimization algorithms.
        the function is implemented here rather than in algorithm classes
        because the function is generic for all algorithms (such as SA and GA),
        and it needs global variables in `RoutingDesigner` to execute. 
        '''
        x.decode()
        freq_dict = {}

        for path in x.path_dict.values():
            for edge in path:
                if edge not in freq_dict:
                    freq_dict[edge] = 0
                freq_dict[edge] += 1
        
        conflicts = list(freq_dict.values())
        return sum(conflicts) / len(conflicts) + max(conflicts) - 1
        # return max(conflicts)

    def run_routing(self) -> None:
        sa = RoutingSimulatedAnnealing(
            self.obj_func, 
            self.rpc,
            T_max=1e-4, 
            T_min=1e-10, 
            L=10, 
            max_stay_counter=1000,
            silent=False
        )
        self.rpc = sa()

    @property
    def routing_result(self) -> RoutingResult:
        self.rpc.decode()
        return RoutingResult(self.layout, self.rpc)
