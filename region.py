import random
import math
import numpy as np
from itertools import combinations as comb
from typing import Dict, List, Tuple, Any, Optional
import matplotlib
from matplotlib import pyplot as plt

color_list = ['red','blue','yellow','purple','green','orange']*10
region_tiles = [3,2,5,3,5,7,8,5,7,3,7,8,3,6,10]
array_w = math.ceil(math.sqrt(sum(region_tiles)))
array_h = math.ceil(math.sqrt(sum(region_tiles)))-1

assert array_w * array_h >= sum(region_tiles)

positions = []
for i in range(array_w):
    for j in range(array_h):
        positions.append((i, j))

pos_idx = list(range(len(positions)))
pos_dict = {i: positions[i] for i in pos_idx}
inv_pos_dict = {positions[i]: i for i in pos_idx}

random.shuffle(pos_idx)

init_x = {}
k = 0
for i, num in enumerate(region_tiles):
    for j in range(num):
        init_x[(i, j)] = pos_idx[k]
        k += 1

DIST = np.zeros([len(positions), len(positions)])

for i, s in enumerate(positions):
    for j, d in enumerate(positions):
        DIST[i, j] = math.sqrt((s[0] - d[0]) ** 2 + (s[1] - d[1]) ** 2)

def obj_func(x):
    total_dist = 0
    for i, num in enumerate(region_tiles):
        for s, d in comb(list(range(num)), 2):
            s = (i, s)
            d = (i, d)
            sid = x[s]
            did = x[d]
            total_dist += DIST[sid, did]
    return total_dist


class SA_Region:

    def __init__(
        self, 
        func, 
        x0, 
        T_max=100, 
        T_min=1e-7, 
        L=300, 
        max_stay_counter=150, 
        **kwargs
    ):
        assert T_max > T_min > 0, 'T_max > T_min > 0'

        self.func = func
        self.T_max = T_max  # initial temperature
        self.T_min = T_min  # end temperature
        self.L = int(L)  # num of iteration under every temperature（also called Long of Chain）
        # stop if best_y stay unchanged over max_stay_counter times (also called cooldown time)
        self.max_stay_counter = max_stay_counter
        self.best_x = init_x

        self.best_y = self.func(self.best_x)
        self.T = self.T_max
        self.iter_cycle = 0
        self.generation_best_X, self.generation_best_Y = [self.best_x], [self.best_y]
        # history reasons, will be deprecated
        self.best_x_history, self.best_y_history = self.generation_best_X, self.generation_best_Y

    def get_new_x(self, x: Dict):
        x_new = x.copy()
        k1, k2 = random.sample(list(x_new.keys()), 2)
        x_new[k1], x_new[k2] = x_new[k2], x_new[k1]
        return x_new

    def cool_down(self):
        self.T = self.T_max / (1 + np.log(1 + self.iter_cycle))

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def run(self):
        x_current, y_current = self.best_x, self.best_y
        stay_counter = 0
        while True:
            # print("temperature: ", self.T, end="\t")
            # print("y_value: ", self.best_y, end="\t")
            # print("stay_counter: ", stay_counter)
            for i in range(self.L):
                x_new = self.get_new_x(x_current)
                y_new = self.func(x_new)

                # Metropolis
                df = y_new - y_current
                if df < 0 or np.exp(-df / self.T) > np.random.rand():
                    x_current, y_current = x_new, y_new
                    if y_new < self.best_y:
                        self.best_x, self.best_y = x_new, y_new

            self.iter_cycle += 1
            self.cool_down()
            self.generation_best_Y.append(self.best_y)
            self.generation_best_X.append(self.best_x)

            # if best_y stay for max_stay_counter times, stop iteration
            if self.isclose(self.best_y_history[-1], self.best_y_history[-2]):
                stay_counter += 1
            else:
                stay_counter = 0

            if self.T < self.T_min:
                stop_code = 'Cooled to final temperature'
                break
            if stay_counter > self.max_stay_counter:
                stop_code = 'Stay unchanged in the last {stay_counter} iterations'.format(stay_counter=stay_counter)
                break

        return self.best_x, self.best_y


def plot_tile(x, y, color='b'):
    w = 0.9
    plt.plot([x,x+w],[y,y],color=color)
    plt.plot([x+w,x+w],[y,y+w],color=color)
    plt.plot([x+w,x],[y+w,y+w],color=color)
    plt.plot([x,x],[y+w,y],color=color)

def plot_result(x: Dict):
    for logic, pos_id in x.items():
        plot_tile(*(positions[pos_id]), color=color_list[logic[0]])

def search_region(
    x: Dict, 
    inv_x: Dict, 
    region_id: int, 
    tile: Tuple, 
    marked: List[int]
):
    tile_id = inv_pos_dict[tile]
    if tile_id in inv_x: 
        logic = inv_x[tile_id]
        if logic[0] != region_id: # belongs to other region
            return
    else: # an idle tile that is not mapped
        return
    
    if tile_id in marked: # already marked
        return
    
    # a valid tile that belongs to the current region
    # a tile that is not marked yet
    # then mark the curren tile
    marked.append(tile_id)

    if tile[0] != 0:
        search_region(x, inv_x, region_id, (tile[0]-1, tile[1]), marked)

    if tile[0] != array_w-1:
        search_region(x, inv_x, region_id, (tile[0]+1, tile[1]), marked)

    if tile[1] != 0:
        search_region(x, inv_x, region_id, (tile[0], tile[1]-1), marked)

    if tile[1] != array_h-1:
        search_region(x, inv_x, region_id, (tile[0], tile[1]+1), marked)

def is_valid(x: Dict) -> bool:
    inv_x = {v: k for k, v in x.items()}
    for i, num in enumerate(region_tiles):
        start_logic = (i, 0)
        start_tile = pos_dict[x[start_logic]]
        marked = []
        search_region(x, inv_x, i, start_tile, marked)
        # print("region_id: ", i, "marked_ratio: ", f"{len(marked)}/{num}", "color: ", color_list[i])
        if len(marked) != num:
            return False
    return True


if __name__ == "__main__":
    cnt = 1
    while True:
        sar = SA_Region( 
            obj_func, 
            init_x, 
            T_max=1e-2, 
            T_min=1e-10, 
            L=10, 
            max_stay_counter=150
        )

        sar.run()
        if is_valid(sar.best_x):
            print(cnt)
        else:
            plot_result(sar.best_x)
            plt.show()
            break
        cnt += 1

    

    

    



