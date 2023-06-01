from maptype import *
import numpy as np
import random

class ClusterSimulatedAnnealing:

    def __init__(
        self, 
        func: Callable[[CIR2PhyIdxMap], float], 
        x0: CIR2PhyIdxMap, 
        T_max: float = 100,
        T_min: float = 1e-7, 
        L: int = 300, 
        max_stay_counter: int = 150,
        silent: bool = False,
        **kwargs
    ) -> None:
        assert T_max > T_min > 0, 'T_max > T_min > 0'

        self.func = func
        self.T_max = T_max  # initial temperature
        self.T_min = T_min  # end temperature
        self.L = int(L)  # num of iteration under every temperature（also called Long of Chain
        
        # stop if best_y stay unchanged over max_stay_counter times (also called cooldown time)
        self.max_stay_counter = max_stay_counter
        self.best_x = x0
        self.silent = silent

        self.best_y = self.func(self.best_x)
        self.T = self.T_max
        self.iter_cycle = 0
        self.generation_best_X, self.generation_best_Y = [self.best_x], [self.best_y]

        # history reasons, will be deprecated
        self.best_x_history, self.best_y_history = self.generation_best_X, self.generation_best_Y

    def get_new_x(self, x: CIR2PhyIdxMap) -> CIR2PhyIdxMap:
        x_new = x.copy()
        k1, k2 = random.sample(list(x_new.keys()), 2)
        x_new[k1], x_new[k2] = x_new[k2], x_new[k1]
        return x_new

    def cool_down(self) -> None:
        self.T = self.T_max / (1 + np.log(1 + self.iter_cycle))

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30) -> bool:
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def run(self) -> Tuple[CIR2PhyIdxMap, float]:
        x_current, y_current = self.best_x, self.best_y
        stay_counter = 0

        while True:
            if not self.silent:
                print("temperature: ", self.T, end="\t")
                print("y_value: ", self.best_y, end="\t")
                print("stay_counter: ", stay_counter)

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
                break

            if stay_counter > self.max_stay_counter:
                break

        return self.best_x, self.best_y