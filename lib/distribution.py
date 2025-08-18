from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import typing as tt
import numpy.typing as npt

import numpy as np


class Distribution(ABC):
    @abstractmethod
    def __add__(self, other: Distribution | int) -> Distribution:
        """Add another distribution or scalar."""
        pass

    @abstractmethod
    def __radd__(self, other: Distribution | int) -> Distribution:
        """Add another distribution or scalar."""
        pass

    @abstractmethod
    def max_with(self, other: Distribution | int) -> Distribution:
        """max(Distribution, other)"""
        pass

    @abstractmethod
    def min_v(self) -> int | float:
        """returns minimal possible value"""
        pass

    @abstractmethod
    def max_v(self) -> int | float:
        """returns maximal possible value"""
        pass

    @abstractmethod
    def __getitem__(self, item: int | float) -> float:
        """returns probability"""
        pass

    @abstractmethod
    def normalize(self) -> None:
        """normalizes distribution"""
        pass

    @abstractmethod
    def e(self) -> float:
        """returns expectation value"""
        pass

    def print_itself(self) -> None:
        """prints distribution itself"""
        pass


class DiscreteDistribution(Distribution):
    def __init__(self, values: npt.NDArray[int], probs: npt.NDArray[float]):
        if probs.shape != values.shape:
            raise ValueError("Probabilities and values must have the same shape.")
        if values.size != len(set(values)):
            raise ValueError("All values should be different must have the same shape.")
        order = values.argsort()
        self.values, self.probs = values[order], probs[order]

    @classmethod
    def _set_from_dict(cls, probs: tt.Dict[int, float]):
        v, p = np.array(list(probs.keys())), np.array(list(probs.values()))
        return DiscreteDistribution(v, p)

    @classmethod
    def set_uniform(cls, min_value: int, max_value: int):
        values = np.array(range(min_value, max_value + 1))
        v_num = max_value + 1 - min_value
        probs = np.full((v_num, ), 1 / v_num)
        return DiscreteDistribution(values, probs)

    def e(self) -> float:
        return float(self.values @ self.probs.T)

    def __add__(self, other: DiscreteDistribution | int) -> DiscreteDistribution:
        if isinstance(other, DiscreteDistribution):
            """Values will be powers of x, probs will be coefficients. 
            We calculate the multiplication of the polynomials"""
            shift = self.values[0]
            coef = np.zeros(self.values[-1] - shift + 1)
            for vi, pi in zip(self.values, self.probs):
                coef[vi - shift] = pi
            shift_other = other.values[0]
            coef_other = np.zeros(other.values[-1] - shift_other + 1)
            for vi, pi in zip(other.values, other.probs):
                coef_other[vi - shift_other] = pi
            coef_res = np.polymul(coef[::-1], coef_other[::-1])[::-1]
            mask = coef_res > 0
            values_res = np.where(mask)[0]
            probs_res = coef_res[mask]
            return DiscreteDistribution(values_res + shift + shift_other, probs_res)
        elif isinstance(other, int):
            return DiscreteDistribution(self.values + other, self.probs)
        else:
            raise TypeError("other in the add function should be one of int or DiscreteDistribution")

    def __radd__(self, other: DiscreteDistribution | int) -> DiscreteDistribution:
        return self.__add__(other)

    def max_with(self, other: DiscreteDistribution | int) -> DiscreteDistribution:
        if isinstance(other, DiscreteDistribution):
            exclude_set = set()
            if other.values[0] < self.values[0]:
                idx = np.searchsorted(other.values, self.values[0], side='left')
                exclude_set = set(other.values[:idx])
            elif other.values[0] > self.values[0]:
                idx = np.searchsorted(self.values, other.values[0], side='left')
                exclude_set = set(self.values[:idx])

            v_set = set(self.values)
            other_v_set = set(other.values)
            res_v_set = set.union(v_set, other_v_set) - exclude_set
            res_values = np.array(list(res_v_set))
            res_values.sort()
            res_probs = np.zeros_like(res_values, dtype=float)

            i = j = 0  # pointers
            p_left = other_p_left = 0.  # probability of hawing less than vi_res
            idx = 0  # res_prob id
            for vi_res in res_values:
                while i < self.values.size and self.values[i] < vi_res:
                    p_left += self.probs[i]
                    i += 1
                while j < other.values.size and other.values[j] < vi_res:
                    other_p_left += other.probs[j]
                    j += 1
                # p, other_p --- probability of having vi_res
                p = self.probs[i] if i < self.values.size and self.values[i] == vi_res else 0.
                other_p = other.probs[j] if j < other.values.size and other.values[j] == vi_res else 0.
                p_res = (p_left * other_p) + (p * other_p_left) + (p * other_p)
                res_probs[idx] = p_res
                idx += 1
            return DiscreteDistribution(res_values, res_probs)
        elif isinstance(other, int):
            idx = np.searchsorted(self.values, other, side='right')
            new_values = np.concat(([other], self.values[idx:]))
            new_probs = np.concat(([self.probs[:idx].sum()], self.probs[idx:]))
            return DiscreteDistribution(new_values, new_probs)
        else:
            raise TypeError("other in the add function should be one of int or DiscreteDistribution")

    def normalize(self, epsilon: float = 1e-8) -> None:
        """Ensure the distribution sums to 1."""
        total = np.sum(self.probs)
        if total != 0:
            self.probs = self.probs / total

        mask = self.probs < epsilon
        self.values = np.delete(self.values, mask)
        self.probs = np.delete(self.probs, mask)

        total = np.sum(self.probs)
        if total != 0:
            self.probs = self.probs / total

    def min_v(self) -> int | float:
        return int(self.values[0])

    def max_v(self) -> int | float:
        return int(self.values[-1])

    def __getitem__(self, item: int) -> float:
        idx = np.searchsorted(self.values, item)
        exists = idx < self.values.size and self.values[idx] == item
        return 0. if not exists else self.probs[idx]

    def print_itself(self):
        print([(int(v), float(p)) for v, p in zip(self.values, self.probs)])
