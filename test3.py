from graphviz import Digraph
from queue import Queue
from copy import deepcopy

class A:
    def __init__(self) -> None:
        self.queue = Queue()

a = A()

a = [1,2,3,4]

a.pop(-1)
print(a)