from maptype import *
from typing import Any, TypeVar
from encoding import LayoutPatternCode, RoutingPatternCode
import numpy as np
from abc import ABCMeta, abstractmethod
import random
from copy import deepcopy, copy

Solution = TypeVar('Solution')

class BaseSimulatedAnnealing(Generic[Solution], Callable, metaclass=ABCMeta):

    def __init__(
        self, 
        func: Callable[[Solution], float], 
        x0: Solution, 
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
        func: Callable[[Solution], float]
            the objective function to be optimized.
            this function must have a scalar-value output
            and the goal is to minimize (not maximize) the function output.
            
        x0: Solution
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
        self.generation_best_Y = [self.best_y]

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=1e-30) -> bool:
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    @abstractmethod
    def update(self, x: Solution) -> None: ...

    @abstractmethod
    def undo_update(self, x: Solution) -> None: ...

    @abstractmethod
    def cool_down(self) -> None: ...

    def __call__(self) -> Solution:
        return self.run()

    def run(self) -> Solution:
        x_current, y_current = self.best_x, self.best_y
        stay_counter = 0

        while True:
            if not self.silent:
                print("temperature: ", self.T, end="\t")
                print("y_value: ", self.best_y, end="\t")
                print("stay_counter: ", stay_counter)

            for i in range(self.L):
                self.update(x_current)
                y_new = self.func(x_current)

                # Metropolis
                df = y_new - y_current
                if df < 0 or np.exp(-df / self.T) > np.random.rand(): # accept new x
                    y_current = y_new
                    if y_new < self.best_y: # record best x
                        self.best_x = deepcopy(x_current)
                        self.best_y = y_new

                else: # discard new x
                    self.undo_update(x_current)

            self.iter_cycle += 1
            self.cool_down()
            self.generation_best_Y.append(self.best_y)

            # if best_y stay for max_stay_counter times, stop iteration
            if self.isclose(self.generation_best_Y[-1], self.generation_best_Y[-2]):
                stay_counter += 1
            else:
                stay_counter = 0

            if self.T < self.T_min:
                break

            if stay_counter > self.max_stay_counter:
                break

        return self.best_x


class LayoutSimulatedAnnealing(BaseSimulatedAnnealing[LayoutPatternCode]):

    def update(self, x: LayoutPatternCode) -> None:
        x.mutation()
    
    def undo_update(self, x: LayoutPatternCode) -> None:
        x.undo_mutation()

    def cool_down(self) -> None:
        self.T = self.T_max / (1 + np.log(1 + self.iter_cycle))


class RoutingSimulatedAnnealing(BaseSimulatedAnnealing[RoutingPatternCode]):

    def update(self, x: RoutingPatternCode) -> None:
        x.mutation()

    def undo_update(self, x: RoutingPatternCode) -> None:
        x.undo_mutation()

    def cool_down(self) -> None:
        self.T = self.T_max / (1 + np.log(1 + self.iter_cycle))