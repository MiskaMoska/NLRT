import random
import queue
import matplotlib
from matplotlib import pyplot as plt
from maptype import *
import networkx as nx
from typing import List, Tuple, Literal
from functools import cached_property
from maptools.core import CTG
from layout_designer import LayoutResult
from acg import ACG

def random_steiner_tree_code(
    term_nodes: List[PhysicalTile],
    all_nodes: List[PhysicalTile]
) -> 'SteinerTreeCode':
    visited_nodes = []
    edge_list, spis = [], []

    remain_nodes = term_nodes.copy()
    init_node = random.choice(term_nodes)
    visited_nodes.append(init_node)
    remain_nodes.remove(init_node)

    for _ in range(len(term_nodes) - 1):
        select_node = random.choice(visited_nodes)
        target_node = random.choice(remain_nodes)
        edge_list.append((select_node, target_node))
        spis.append(random.choice([True, False]))

        remain_nodes.remove(target_node)
        visited_nodes.append(target_node)

    return SteinerTreeCode(
        edge_list, spis,
        random.choice(term_nodes),
        term_nodes, all_nodes
    )


class SteinerTreeCode(nx.Graph):

    def __init__(
        self, 
        edges: List[ArbitaryEdge], 
        spis: List[bool], 
        root: PhysicalTile,
        term_nodes: List[PhysicalTile],
        all_nodes: List[PhysicalTile],
        *args, **kwargs
    ) -> None:
        '''
        Encoding Data Structure of Steiner Tree
        
        Parameters
        ----------
        edges: List[ArbitaryEdge]
            edge list of the spanning tree. 
            spanning tree is the base structure for steiner tree, the steiner tree
            is built according to the spanning tree edges, the steiner point 
            indicators, and the given root node. 
            the `SteinerTreeCode` itself is actually a graph of the spanning tree.
            running the `SteinerTreeCode.decode` method results in the automatical 
            construction of the corresponding steiner tree.

        spis: List[bool]
            steiner point indicators.
            determines whether to use XY routing or YX routing when building raw
            steiner trees based on the spanning tree edges. 

        root: PhysicalTile
            root node of the tree, for DFS and BFS.
            under different root node placements, it can generate different steiner 
            tree structures for DFS and BFS.

        term_nodes: List[PhysicalTile]
            terminal nodes, including source node (sender) and sink nodes (receivers).
            note that source node is not equivalent to root node, if the source node
            is not the root, it can also successfully send messages to the sink nodes
            following the edges of the steiner tree.
            the root node is nothing but a concept for DFS or BFS searching.

        all_nodes: List[PhysicalTile]
            all nodes in network on chip, it must cover the hanan plane for the given
            terminal nodes, it describes the constrained space that the multicast can 
            be routed to, any node in `all_nodes` can be a routing node for multicast.
        '''
        if len(edges) == 0:
            raise ValueError("got empty edge list")
        
        if len(edges) != len(spis):
            raise ValueError("got edge list length not match spi list")

        if root not in term_nodes:
            raise RuntimeError(
                f"root {root} not in terminal nodes: {term_nodes}")

        self.root = root
        self.all_nodes = all_nodes
        self.term_nodes = term_nodes
        
        super().__init__(*args, **kwargs)
        self.add_nodes_from(all_nodes)

        for (edge, spi) in zip(edges, spis):
            self.add_edge(*edge, spi=spi)

        self.node_pos = {n: n for n in self.all_nodes}
        self.node_color = {n: 'red' if n in self.term_nodes 
                            else 'black' for n in self.all_nodes}
        self.node_color[self.root] = 'green'

    def mutation(self) -> None:
        method = random.choice([True, False])
        method = True
        print(f"mutation method: ", "edge replacement" if method else "root relocation")

        if method: # replace edge
            edge = random.choice(list(self.edges))
            self.remove_edge(*edge) # remove an edge randomly

            part1 = nx.node_connected_component(self, self.root)
            part2 = set(self.term_nodes) - part1
            node1 = random.choice(list(part1))
            node2 = random.choice(list(part2))

            spi = random.choice([True, False])
            self.add_edge(node1, node2, spi=spi)
        
        else: # relocate root
            while True:
                r = random.choice(self.term_nodes)
                if r != self.root: break

            self.node_color[self.root] = 'red'
            self.node_color[r] = 'green'
            self.root = r

    def decode(self) -> nx.Graph:
        self._decode_to_raw_steiner()
        self._decode_to_true_steiner(method='dfs')
        self._decode_to_true_steiner(method='bfs')
        return self.tstg_b

    def _decode_to_raw_steiner(self) -> None:
        self.rstg = nx.Graph() # raw steiner tree graph
        self.rstg.add_nodes_from(self.all_nodes)
        for edge in self.edges:
            edge_data = self.get_edge_data(*edge)
            spi = edge_data['spi']
            self._add_steiner_route(self.rstg, edge, spi)

    @staticmethod
    def _add_steiner_route(
        rstg: nx.Graph,
        spanning_edge: ArbitaryEdge,
        spi: bool
    ) -> None:
        src, dst = spanning_edge
        cur = src

        while True:
            if cur == dst:
                return
            st = [0, 0]
            i = 1 if spi else 0

            if cur[i] == dst[i]:
                st[1-i] = 1 if dst[1-i] > cur[1-i] else -1
            else:
                st[i] = 1 if dst[i] > cur[i] else -1
            nxt = (cur[0] + st[0], cur[1] + st[1])

            rstg.add_edge(cur, nxt)
            cur = nxt

    def _decode_to_true_steiner(
        self, 
        method: Literal['dfs', 'bfs'] = 'dfs'
    ) -> None:
        if method == 'dfs':
            self.tstg_d = nx.Graph() # true steiner tree graph by dfs
            self.tstg_d.add_nodes_from(self.all_nodes)
            visited_nodes, remain_edges = [], []
            self._dfs(self.rstg, self.root, visited_nodes, remain_edges)
            self.tstg_d.add_edges_from(remain_edges)

        elif method == 'bfs':
            self.tstg_b = nx.Graph() # true steiner tree graph by bfs
            self.tstg_b.add_nodes_from(self.all_nodes)
            visited_nodes = []
            self._bfs(self.rstg, self.tstg_b, self.root, visited_nodes)
    
    def _dfs(
        self,
        g: nx.Graph, 
        cur: Tuple, 
        visited: List[Tuple], 
        remain: List[Tuple]
    ) -> bool:
        visited.append(cur)
        flag, leaf = False, True
        for nxt in g.neighbors(cur):
            if nxt not in visited:
                remain.append((cur, nxt))
                if self._dfs(g, nxt, visited, remain): # (cur, nxt) is not a leaf branch
                    flag = True
                else: # (cur, nxt) is a leaf branch
                    remain.remove((cur, nxt))
            else:
                leaf = False

        if cur in self.term_nodes:
            return True
        else: 
            return leaf or flag

    def _bfs(
        self,
        g: nx.Graph,
        tg: nx.Graph,
        start: Tuple, 
        visited: List[Tuple]
    ) -> None:          
        fifo = queue.Queue()  
        rethink_fifo = queue.Queue()           
        fifo.put(start)           
        visited.append(start)

        while True:
            if fifo.empty() and rethink_fifo.empty():
                break

            if not fifo.empty():
                node = fifo.get()
                endnode = True
                for nxt in g.neighbors(node):
                    if nxt not in visited:
                        fifo.put(nxt)
                        visited.append(nxt)
                        tg.add_edge(node, nxt)
                        endnode = False
                    
                if endnode and (node not in self.term_nodes):
                    pred = list(tg.neighbors(node))[0]
                    tg.remove_edge(pred, node)
                    rethink_fifo.put(pred)
            
            if not rethink_fifo.empty():
                renode = rethink_fifo.get()
                if tg.degree(renode) == 1 and renode not in self.term_nodes:
                    repred = list(tg.neighbors(renode))[0]
                    tg.remove_edge(repred, renode)
                    rethink_fifo.put(repred)

    def draw_graph(self, graph: nx.Graph, ax: plt.Axes) -> None:
        nx.draw(
            graph,
            self.node_pos, 
            node_size=40, 
            width=1, 
            node_color=list(self.node_color.values()),
            ax=ax
        )

    @cached_property
    def reroot_possibility(self) -> float:
        pass

    def draw4(self, ax1, ax2, ax3, ax4) -> None:
        self.draw_graph(self, ax1)
        self.draw_graph(self.rstg, ax2)
        self.draw_graph(self.tstg_b, ax3)
        self.draw_graph(self.tstg_d, ax4)

        ax1.set_title('spanning tree', y=-0.1)
        ax2.set_title('raw steiner', y=-0.1)
        ax3.set_title('steiner bfs', y=-0.1)
        ax4.set_title('steiner dfs', y=-0.1)


class RoutingPatternCode():

    def __init__(self, ctg: CTG, acg: ACG, layout: LayoutResult) -> None:
        self.noc_w = acg.w
        self.noc_h = acg.h
        self.all_nodes = acg.nodes

        self.stc_dict: Dict[str, SteinerTreeCode] = {} # steiner tree code dict
        self.src_dict: Dict[str, PhysicalTile] = {} # src node dict
        self.sid_dict: Dict[str, int] = {} # stream ID dict
        self.path_dict: Dict[str, List[MeshEdge]] = {} # communication path dict

        for sid, (c, src, dst) in enumerate(ctg.cast_trees):
            physrc = layout[src]
            phydst = [layout[d] for d in dst]
            term_nodes = phydst + [physrc]
            self.stc_dict[c] = random_steiner_tree_code(
                term_nodes, self.all_nodes
            )
            self.src_dict[c] = physrc
            self.sid_dict[c] = sid

    def decode(self) -> None:
        for comm, stc in self.stc_dict.items():
            tstg: nx.Graph = stc.decode()
            tree: nx.DiGraph = nx.bfs_tree(tstg, self.src_dict[comm])
            self.path_dict[comm] = list(tree.edges)
            