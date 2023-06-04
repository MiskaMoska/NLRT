import random
from maptools.core import CTG, LogicalTile, PhysicalTile
from acg import ACG
import numpy as np
from graphviz import Graph as ZGraph
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple, Literal
from maptype import CIRTile

class LayoutResult(object):

    def __init__(
        self,
        noc_w: int,
        noc_h: int,
        hotmap: Dict[CIRTile, int],
        log_dict: Dict[CIRTile, LogicalTile],
        phy_dict: Dict[int, PhysicalTile]
    ) -> None:
        self.noc_w = noc_w
        self.noc_h = noc_h
        self.hotmap = hotmap
        self.log_dict = log_dict
        self.phy_dict = phy_dict
        self._prepare_tile_color()

        self.l2p_map = (
            {log_dict[cir]:  phy_dict[hotmap[cir]] 
            for cir in hotmap.keys()}
        )
        
    def __getitem__(self, log_tile: LogicalTile) -> PhysicalTile:
        return self.l2p_map[log_tile]

    def _prepare_tile_color(self) -> None:
        dark_colors = [
            color for _, color in mcolors.CSS4_COLORS.items() 
            if all(c <= 0.7 for c in mcolors.to_rgb(color))
        ]
        k = len(self.hotmap) // len(dark_colors) + 1
        self.colors = (dark_colors * k)[:len(self.hotmap)]
    
    def draw(self, engine: Literal['fdp', 'mplt'] = 'fdp') -> None:
        # draw through matplotlib
        if engine == 'mplt': 
            plt.figure(figsize=(self.noc_w, self.noc_h))
            for cir, pidx in self.hotmap.items():
                phytile = self.phy_dict[pidx]
                self._draw_tile_mplt(phytile, cir[0], self.colors[cir[0]])

            plt.show()

        # draw through graphviz fdp
        elif engine == 'fdp':
            fdp = ZGraph('layout', engine='fdp', format='pdf')
            self.draw_fdp(fdp)
            fdp.render(cleanup=True, directory='.', view=True)

    @staticmethod
    def _draw_tile_mplt(
        phytile: PhysicalTile, 
        cid: int, 
        color: str
    ) -> None:
        w = 0.8
        x, y = phytile
        y = -y
        linewidth = 4

        plt.plot([x,x+w], [y,y], color=color, linewidth=linewidth)
        plt.plot([x+w,x+w], [y,y+w], color=color, linewidth=linewidth)
        plt.plot([x+w,x], [y+w,y+w], color=color, linewidth=linewidth)
        plt.plot([x,x], [y+w,y], color=color, linewidth=linewidth)
        
        plt.text(
            x + w/2, y + w/2,
            'C'+str(cid),
            fontsize=15,
            verticalalignment='center', 
            horizontalalignment='center', 
            color=color,
            fontweight='bold'
        )

    def draw_fdp(
        self, fdp: ZGraph,
        width: str = '0.8',
        penwidth: str = '5',
        fontsize: str ='24',
        dist: int = 1
    ) -> None:
        for cir, pidx in self.hotmap.items():
            phytile = self.phy_dict[pidx]
            pos = f'{phytile[0] * dist},{-phytile[1] * dist}!'
            fdp.node(
                str(phytile), 
                f'C{cir[0]}', 
                color=self.colors[cir[0]],
                fontcolor=self.colors[cir[0]],
                pos=pos,
                shape='square',
                width=width,
                penwidth=penwidth,
                fontsize=fontsize,
                fontname='Arial'
            )