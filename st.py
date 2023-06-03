import networkx as nx
import random
import matplotlib
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple, Literal
import queue

W = 10
H = 10
T = 30

candidate_nodes = [(x, y) for x in range(W) for y in range(H)]
node_pos = {(x, y) : (x * 1, y * 1) for x in range(W) for y in range(H)}
terminal_nodes = random.sample(candidate_nodes, T)
ORIGIN_NODE_COLOR = ['red' if node in terminal_nodes else 'black' for node in candidate_nodes]

def random_spanning_tree(
    candidate_nodes: List, 
    terminal_nodes: List
) -> nx.Graph:
    rc = nx.Graph()
    rc.add_nodes_from(candidate_nodes)
    node_num = len(terminal_nodes)

    visited_nodes = []
    remain_nodes = terminal_nodes.copy()

    init_node = random.choice(terminal_nodes)
    visited_nodes.append(init_node)
    remain_nodes.remove(init_node)
    edge_cnt = 0

    while True:
        select_node = random.choice(visited_nodes)
        target_node = random.choice(remain_nodes)
        
        rc.add_edge(select_node, target_node)
        remain_nodes.remove(target_node)
        visited_nodes.append(target_node)
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


def reduce(
    g: nx.Graph, 
    terminal_nodes: List, 
    cut: bool = True, 
    method: Literal['dfs', 'bfs'] = 'dfs'
) -> nx.Graph:

    def dfs(
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
                if dfs(g, nxt, visited, remain): # (cur, nxt) is not a leaf branch
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

    # def bfs(
    #     g: nx.Graph, 
    #     start: Tuple, 
    #     visited: List[Tuple], 
    #     remain: List[Tuple]
    # ) -> None:          
    #     fifo = queue.Queue()             
    #     fifo.put(start)           
    #     visited.append(start)

    #     while not fifo.empty():
    #         node = fifo.get()
    #         for nxt in g.neighbors(node):
    #             if nxt not in visited:
    #                 fifo.put(nxt)
    #                 visited.append(nxt)
    #                 remain.append((node, nxt))

    def bfs(
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
                    
                if endnode and (node not in terminal_nodes):
                    pred = list(tg.neighbors(node))[0]
                    tg.remove_edge(pred, node)
                    NODE_COLOR[candidate_nodes.index(node)] = 'green'
                    rethink_fifo.put(pred)
            
            if not rethink_fifo.empty():
                renode = rethink_fifo.get()
                if tg.degree(renode) == 1 and renode not in terminal_nodes:
                    repred = list(tg.neighbors(renode))[0]
                    tg.remove_edge(repred, renode)
                    NODE_COLOR[candidate_nodes.index(renode)] = 'green'
                    rethink_fifo.put(repred)

    rc = nx.Graph()
    nodes = list(g.nodes)
    rc.add_nodes_from(nodes)
    source = random.choice(terminal_nodes)
    visited_nodes, remain_edges = [], []
    if method == 'dfs':
        dfs(g, source, visited_nodes, remain_edges)
    else:
        bfs(g, rc, source, visited_nodes)
    rc.add_edges_from(remain_edges)
    return rc

cnt = 0
while True:
    cnt += 1
    print(cnt)

    plt.close()
    NODE_COLOR = ['red' if node in terminal_nodes else 'black' for node in candidate_nodes]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
    rst = random_spanning_tree(candidate_nodes, terminal_nodes)
    edges = list(rst.edges)
    stt = build_steiner_tree(candidate_nodes, edges) 
    rsttb = reduce(stt, terminal_nodes, cut=True, method='bfs')
    rsttd = reduce(stt, terminal_nodes, cut=True, method='dfs')

    flag = False
    for node in rsttb.nodes:
        if rsttb.degree(node) == 1 and node not in terminal_nodes:
            flag = True

    if flag or True:
        nx.draw(
            rst,
            node_pos, 
            node_size=40, 
            width=1, 
            # arrowsize=5, 
            node_color=ORIGIN_NODE_COLOR,
            # arrowstyle='-|>'
            ax=ax1
        )

        nx.draw(
            stt,
            node_pos, 
            node_size=40, 
            width=1, 
            # arrowsize=5, 
            node_color=ORIGIN_NODE_COLOR,
            # arrowstyle='-|>'
            ax=ax2
        )

        nx.draw(
            rsttb,
            node_pos, 
            node_size=40, 
            width=1, 
            # arrowsize=5, 
            node_color=NODE_COLOR,
            # arrowstyle='-|>'
            ax=ax3
        )

        nx.draw(
            rsttd,
            node_pos, 
            node_size=40, 
            width=1, 
            # arrowsize=5, 
            node_color=ORIGIN_NODE_COLOR,
            # arrowstyle='-|>'
            ax=ax4
        )
        ax1.set_title('spanning tree', y=-0.1)
        ax2.set_title('raw steiner', y=-0.1)
        ax3.set_title('steiner bfs', y=-0.1)
        ax4.set_title('steiner dfs', y=-0.1)
        plt.show()