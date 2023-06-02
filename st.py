import networkx as nx
import random
import matplotlib
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple

candidate_nodes = [(x, y) for x in range(8) for y in range(8)]
node_pos = {(x, y) : (x * 1, y * 1) for x in range(8) for y in range(8)}
terminal_nodes = random.sample(candidate_nodes, 32)

def random_spanning_tree(
    candidate_nodes: List, 
    terminal_nodes: List
) -> nx.Graph:
    rc = nx.Graph()
    rc.add_nodes_from(candidate_nodes)
    node_num = len(terminal_nodes)

    marked_nodes = []
    remain_nodes = terminal_nodes.copy()

    init_node = random.choice(terminal_nodes)
    marked_nodes.append(init_node)
    remain_nodes.remove(init_node)
    edge_cnt = 0

    while True:
        select_node = random.choice(marked_nodes)
        target_node = random.choice(remain_nodes)
        
        rc.add_edge(select_node, target_node)
        remain_nodes.remove(target_node)
        marked_nodes.append(target_node)
        edge_cnt += 1
        if edge_cnt == node_num - 1:
            return rc

def add_steiner_route(
    g: nx.Graph,
    spanning_edge: Tuple,
    steiner_indicator: bool
) -> None:
    src = spanning_edge[0]
    dst = spanning_edge[1]
    cur = src

    while True:
        if cur == dst:
            return
        
        st = [0, 0]
        i = 1 if steiner_indicator else 0

        if cur[i] == dst[i]:
            st[1-i] = 1 if dst[1-i] > cur[1-i] else -1
        else:
            st[i] = 1 if dst[i] > cur[i] else -1
        nxt = (cur[0] + st[0], cur[1] + st[1])

        g.add_edge(cur, nxt)
        cur = nxt


def build_steiner_tree(
    candidate_nodes: List,
    spanning_edges: List
) -> nx.Graph:
    rc = nx.Graph()
    rc.add_nodes_from(candidate_nodes)
    for edge in spanning_edges:
        si = random.choice([True, False])
        add_steiner_route(rc, edge, si)

    return rc


def reduce(g: nx.Graph, terminal_nodes: List, cut: bool = True) -> nx.Graph:

    def dfs(g: nx.Graph, cur: Tuple, marked: List[Tuple], remain: List[Tuple]) -> bool:
        marked.append(cur)
        flag, leaf = False, True
        for nxt in g.neighbors(cur):
            if nxt not in marked:
                remain.append((cur, nxt))
                if dfs(g, nxt, marked, remain): # (cur, nxt) is not a leaf branch
                    flag = True
                else: # (cur, nxt) is a leaf branch
                    if cut:
                        remain.remove((cur, nxt))
            else:
                leaf = False

        if cur in terminal_nodes:
            return True
        else: 
            return leaf or flag

    rc = nx.Graph()
    nodes = list(g.nodes)
    rc.add_nodes_from(nodes)
    source = random.choice(terminal_nodes)
    marked_nodes, remian_edges = [], []
    dfs(g, source, marked_nodes, remian_edges)
    rc.add_edges_from(remian_edges)
    return rc

while True:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    rst = random_spanning_tree(candidate_nodes, terminal_nodes)
    edges = list(rst.edges)
    stt = build_steiner_tree(candidate_nodes, edges) 
    rstt = reduce(stt, terminal_nodes, cut=True)
    rsttc = reduce(stt, terminal_nodes)
    node_color = ['red' if node in terminal_nodes else 'black' for node in candidate_nodes]


    nx.draw(
        rst,
        node_pos, 
        node_size=40, 
        width=1, 
        # arrowsize=5, 
        node_color=node_color,
        # arrowstyle='-|>'
        ax=ax1
    )

    nx.draw(
        stt,
        node_pos, 
        node_size=40, 
        width=1, 
        # arrowsize=5, 
        node_color=node_color,
        # arrowstyle='-|>'
        ax=ax2
    )

    nx.draw(
        rstt,
        node_pos, 
        node_size=40, 
        width=1, 
        # arrowsize=5, 
        node_color=node_color,
        # arrowstyle='-|>'
        ax=ax3
    )

    plt.show()