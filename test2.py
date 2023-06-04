'''
test the locality of mutation operation for STC
'''

from encoding import *

W = 7
H = 7
T = 23

candidate_nodes = [(x, y) for x in range(W) for y in range(H)]
node_pos = {(x, y) : (x * 1, y * 1) for x in range(W) for y in range(H)}
terminal_nodes = random.sample(candidate_nodes, T)

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

rst = random_spanning_tree(candidate_nodes, terminal_nodes)
spis = [random.choice([True, False]) for i in range(len(terminal_nodes)-1)]

stc = SteinerTreeCode(
    list(rst.edges), 
    spis, 
    random.choice(terminal_nodes), 
    terminal_nodes,
    candidate_nodes
)

while True:
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    stc.decode()
    stc.draw4(*(ax[0, :]))
    stc.mutation()
    stc.decode()
    stc.draw4(*(ax[1, :]))
    plt.show()
    plt.close()