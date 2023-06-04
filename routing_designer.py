import random
from maptools.core import CTG, LogicalTile, PhysicalTile
from acg import ACG
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple, Any
from maptype import CIRTile, CIR2PhyIdxMap, Logical2PhysicalMap
from functools import cached_property
from itertools import combinations as comb
from csa import ClusterSimulatedAnnealing
from layout_designer import LayoutResult
from encoding import RoutingPatternCode
from graphviz import Digraph as ZDigraph

__all__ = ['RoutingDesigner', 'RoutingResult']

class RoutingDesigner(object):

    def __init__(self, ctg: CTG, acg: ACG, layout: LayoutResult) -> None:
        self.noc_w = acg.w
        self.noc_h = acg.h
        self.layout = layout

        self.hotrtg = RoutingPatternCode(ctg, acg, layout)

    @property
    def routing_result(self) -> 'RoutingResult':
        self.hotrtg.decode()
        return RoutingResult(self.layout, self.hotrtg)


class RoutingResult(object):
    
    def __init__(self, layout: LayoutResult, rpc: RoutingPatternCode) -> None:
        self.layout = layout
        self.path_dict = rpc.path_dict
        self.src_dict = rpc.src_dict
        self.sid_dict = rpc.sid_dict
        self._prepare_route_color()

    def __getitem__(self, comm: str) -> Dict[str, Any]:
        return {
            'sid': self.sid_dict[comm],
            'src': self.src_dict[comm],
            'path': self.path_dict[comm]
        }

    def _prepare_route_color(self) -> None:
        dark_colors = [
            color for _, color in mcolors.CSS4_COLORS.items() 
            if all(c <= 0.6 for c in mcolors.to_rgb(color))
        ]
        k = len(self.path_dict) // len(dark_colors) + 1
        self.colors = (dark_colors * k)[:len(self.path_dict)]

    def draw(self) -> None:
        fdp = ZDigraph('routing', engine='fdp', format='pdf')
        # draw layout tiles
        self.layout.draw_fdp(
            fdp,
            width='1',
            penwidth='5',
            fontsize='24',
            dist=1.8
        )
        # draw routing paths
        for i, path in enumerate(self.path_dict.values()):
            for edge in path:
                fdp.edge(
                    str(edge[0]),
                    str(edge[1]),
                    color=self.colors[i],
                    penwidth='3'
                )
        fdp.render(cleanup=True, directory='.', view=True)


        
