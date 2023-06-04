from graphviz import Digraph
import matplotlib.colors as mcolors
dot = Digraph('graph', engine='fdp', format='pdf')
# dot.attr(rankdir='LR')
for name, color in mcolors.CSS4_COLORS.items():
    color = color
    print(color)
    break

# plot nodes
dot.node('1', pos='0,0!', color=color)
dot.node('2', pos='1,0!')
dot.node('3', pos='2,0!')
dot.node('4', pos='0,1!')
dot.node('5', pos='1,1!')
dot.edge('1','2')
dot.edge('1','2')
dot.edge('2','3')
dot.edge('4','5')
dot.edge('5','4')
dot.edge('5','4')
dot.edge('4','1')

dot.render(cleanup=True, directory='.', view=True)