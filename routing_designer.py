import random
from maptools.core import CTG, LogicalTile, PhysicalTile
from acg import ACG
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple
from maptype import CIRTile, CIR2PhyIdxMap, Logical2PhysicalMap
from functools import cached_property
from itertools import combinations as comb
from csa import ClusterSimulatedAnnealing

__all__ = ['RoutingDesigner']

class RoutingDesigner(object):

    def __init__(self, ctg: CTG, acg: ACG) -> None:
        self.noc_w = acg.w
        self.noc_h = acg.h

    def 
        
