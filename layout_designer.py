import random
from maptools.core import CTG, LogicalTile, PhysicalTile
from acg import ACG
import numpy as np
from graphviz import Graph as ZGraph
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple, Literal
from maptype import CIRTile, CIR2PhyIdxMap, Logical2PhysicalMap
from functools import cached_property
from itertools import combinations as comb
from csa import ClusterSimulatedAnnealing

__all__ = ['LayoutDesigner', 'LayoutResult']

class LayoutDesigner(object):

    def __init__(self, ctg: CTG, acg: ACG) -> None:
        '''
        Tile-NoC Layout Designer
        Determines the 1-to-1 mapping between logical tile and physical tiles

        Parameter
        ---------
        ctg: CTG
            Communication Trace Graph of the AI task.

        acg: ACG
            Architecture Characterization Graph of the NoC.
        '''
        # A list of all logical tiles
        self.log_tiles: List[LogicalTile]

        # A list of all possible physical tiles
        self.phy_tiles: List[PhysicalTile] 

        # A list of the number of tiles in each cluster
        self.cluster_list: List[int] 

        # A list of cluster-index-represented (CIR) tiles
        self.cir_tiles: List[CIRTile] 

        # A list of indices of all physical tile
        self.phy_indices: List[int] 

        # A dictionary with CIR tile as keys and physical tile indices as values
        # This variable is used to hold the initial and optimal layout pattern
        self.hotmap: Dict[CIRTile, int] 

        # A dictionary with CIR tile as keys and logical tile as values
        self.log_dict: Dict[CIRTile, LogicalTile] 

        # A dictionary with physical tile indices as keys and physical tiles as values
        self.phy_dict: Dict[int, PhysicalTile]

        # A dictionary with physical tile as keys and physical tile indices as values
        # This is the inverted-mapped version of `self.phy_dict`
        self.inv_phy_dict:  Dict[PhysicalTile, int]

        self.noc_w = acg.w
        self.noc_h = acg.h

        self.acg_nodes = acg.nodes
        self.log_tiles = ctg.tile_nodes
        assert len(self.acg_nodes) >= len(self.log_tiles), "need larger NoC"

        self.ctg_clusters = list(ctg.clusters)
        self.phy_tiles = self.acg_nodes
        self.phy_indices = list(range(len(self.acg_nodes)))

        self.phy_dict = {i: n for i, n in enumerate(self.acg_nodes)}
        self.inv_phy_dict = {n: i for i, n in enumerate(self.acg_nodes)}

        self.cluster_list = []
        self.cir_tiles = []
        self.log_dict = {}

        for i, (_, tiles) in enumerate(self.ctg_clusters):
            self.cluster_list.append(len(tiles))
            for k, tile in enumerate(tiles):
                cir_tile = (i, k)
                self.cir_tiles.append(cir_tile)
                self.log_dict[cir_tile] = tile

        self.hotmap = {}
        self.init_layout()

    def init_layout(self) -> None:
        '''
        This method re-initializes the layout pattern by
        resetting the hotmap to random values.
        Call this method before launching a new round of optimization.
        '''
        temp_phy_indices = self.phy_indices.copy()
        random.shuffle(temp_phy_indices)
        for i, cir in enumerate(self.cir_tiles):
            self.hotmap[cir] = temp_phy_indices[i]

    @cached_property
    def ptdm(self) -> np.ndarray:
        '''
        physical tile distance matrix (PTDM)
        '''
        res = np.zeros([len(self.acg_nodes)]*2)
        for i, s in enumerate(self.acg_nodes):
            for j, d in enumerate(self.acg_nodes):

                # using the Manhattan distance
                res[i, j] = abs(s[0] - d[0]) + abs(s[1] - d[1])

        return res

    def obj_func(self, x: CIR2PhyIdxMap) -> float:
        total_dist = 0
        for i, num in enumerate(self.cluster_list):
            for s, d in comb(list(range(num)), 2):
                s_cir, d_cir = (i, s), (i, d)
                s_pidx, d_pidx = x[s_cir], x[d_cir]
                total_dist += self.ptdm[s_pidx, d_pidx]

        return total_dist

    def search_cluster(
        self,
        x: CIR2PhyIdxMap, 
        inv_x: Dict[int, CIRTile], 
        cluster_id: int, 
        tile: PhysicalTile, 
        marked: List[int]
    ) -> None:
        '''
        This method searched through a cluster and marks physical tiles 
        as many as possible following a patch-manner, if the number of 
        marked tiles equals the number of tiles in the current cluster, 
        it means the current cluster is mapped to a patch, otherwise, 
        it is not mapped to a patch.

        Parameters
        ----------
        x: CIR2PhyIdxMap
            the layout pattern

        inv_x: Dict[int, CIRTile]
            the inverted-mapped version of `x`

        cluster_id: int
            the cluster ID of the current cluster being searched
        
        tile: PhysicalTile
            the current physical tile being searched

        marked: List[int]
            a list of the physical tile indices that have been marked
        '''
        pidx = self.inv_phy_dict[tile]
        if pidx in inv_x: 
            logic = inv_x[pidx]
            if logic[0] != cluster_id: # belongs to other cluster
                return
        else: # an idle tile that is not mapped
            return
        
        if pidx in marked: # already marked
            return
        
        # If a physical tile meets three requirements:
        # 1. A tile that has been mapped
        # 2. A tile belonging to the current cluster
        # 3. A tile that is not marked yet
        # then current physical tile should be marked
        marked.append(pidx)

        if tile[0] != 0:
            self.search_cluster(x, inv_x, cluster_id, (tile[0]-1, tile[1]), marked)

        if tile[0] != self.noc_w-1:
            self.search_cluster(x, inv_x, cluster_id, (tile[0]+1, tile[1]), marked)

        if tile[1] != 0:
            self.search_cluster(x, inv_x, cluster_id, (tile[0], tile[1]-1), marked)

        if tile[1] != self.noc_h-1:
            self.search_cluster(x, inv_x, cluster_id, (tile[0], tile[1]+1), marked)

    def is_patches(self, x: CIR2PhyIdxMap) -> bool:
        '''
        This method checks the valadity of the given layout pattern,
        that is, whether all clusters are mapped to a patch region.
        '''
        inv_x = {v: k for k, v in x.items()}

        for cluster_id, num in enumerate(self.cluster_list):
            start_cir = (cluster_id, 0)
            start_phy_tile = self.phy_dict[x[start_cir]]

            marked = []
            self.search_cluster(x, inv_x, cluster_id, start_phy_tile, marked)
            if len(marked) != num:
                print(f'non-patch detected at cluster {cluster_id}')
                return False
        return True

    def run_layout(self) -> None:
        sa = ClusterSimulatedAnnealing(
            self.obj_func, 
            self.hotmap,
            T_max=1e-2, 
            T_min=1e-10, 
            L=10, 
            max_stay_counter=150,
            silent=True  
        )
        sa.run()
        self.hotmap = sa.best_x
        print(f"is_patches: {self.is_patches(self.hotmap)}")

    @property
    def layout_result(self) -> 'LayoutResult':
        return LayoutResult(
            self.noc_w,
            self.noc_h,
            self.hotmap,
            self.log_dict,
            self.phy_dict
        )


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
