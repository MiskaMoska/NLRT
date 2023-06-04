from maptype import *
from typing import Any
from encoding import RoutingPatternCode
import numpy as np
from abc import ABCMeta, abstractmethod
import random

class BaseSimulatedAnnealing(Callable, metaclass=ABCMeta):

    def __init__(
        self, 
        func: Callable, 
        x0: Any, 
        T_max: float = 100,
        T_min: float = 1e-7, 
        L: int = 300, 
        max_stay_counter: int = 150,
        silent: bool = False,
        *args, **kwargs
    ) -> None:
        '''
        Base class for Simulated Annealing
        This class is borrowed from the version of scikit-opt
        https://github.com/guofei9987/scikit-opt/blob/master/sko/SA.py
        with some application-specific modifications.
        
        Parameters
        ----------
        func: Callable
            the objective function to be optimized.
            this function must have a scalar-value output
            and the goal is to minimize (not maximize) the function output.
            
        x0: Any
            initial solution, must be the same type with the input of `func`.

        T_max: float
            maximum or initial temperature.

        T_min: float
            minimum or final temperature.

        L: int
            number of iterations under every temperature.

        max_stay_counter: int
            invariance counter threshold.

        silent: bool
            whether to run without logging to the terminal.
        '''
        super().__init__(*args, **kwargs)

        self.func = func
        assert T_max > T_min > 0, 'T_max > T_min > 0'
        self.T_max = T_max # initial temperature
        self.T_min = T_min # end temperature
        self.L = int(L) # num of iteration under every temperature（also called Long of Chain
        
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

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30) -> bool:
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    @abstractmethod
    def get_new_x(self, x: Any) -> Any: ...

    @abstractmethod
    def cool_down(self) -> None: ...

    def __call__(self) -> Tuple[Any, float]:
        return self.run()

    def run(self) -> Any:
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

        return self.best_x


class LayoutSimulatedAnnealing(BaseSimulatedAnnealing):

    def get_new_x(self, x: CIR2PhyIdxMap) -> CIR2PhyIdxMap:
        x_new = x.copy()
        k1, k2 = random.sample(list(x_new.keys()), 2)
        x_new[k1], x_new[k2] = x_new[k2], x_new[k1]
        return x_new

    def cool_down(self) -> None:
        self.T = self.T_max / (1 + np.log(1 + self.iter_cycle))


class RoutingSimulatedAnnealing(BaseSimulatedAnnealing):

    def get_new_x(self, x: RoutingPatternCode) -> RoutingPatternCode:
        comm = random.choice(x.comms) # randomly choose a communication
        stc = x.stc_dict[comm] # get the handler the corresponding STC
        stc.mutation() # perform mutation on the STC
        return x

    def cool_down(self) -> None:
        self.T = self.T_max / (1 + np.log(1 + self.iter_cycle))