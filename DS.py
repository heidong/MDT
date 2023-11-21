from __future__ import print_function
from itertools import chain, combinations
from functools import partial, reduce
from operator import mul
from math import log, fsum, sqrt
from random import random, shuffle, uniform
import sys

try:
    import numpy
    try:
        from scipy.stats import chi2
        from scipy.optimize import fmin_cobyla
    except:
        print('SciPy not found: some features will not work.', file=sys.stderr)
except:
    print('NumPy not found: some features will not work.', file=sys.stderr)
    
   


class BaseMassFunction(dict):
    def __init__(self, source=None):
        """
        Creates a new mass function.
        
        If 'source' is not None, it is used to initialize the mass function.
        It can either be a dictionary mapping hypotheses to non-negative mass values
        or an iterable containing tuples consisting of a hypothesis and a corresponding mass value.
        """
        if source != None:
            if isinstance(source, dict):
                source = source.items()
            for (h, v) in source:
                self[h] += v

    @staticmethod
    def _convert(hypothesis):
        #print(hypothesis)
        """Convert hypothesis to a 'frozenset' in order to make it hashable."""
        if isinstance(hypothesis, frozenset):
            return hypothesis
        else:
         #   print(frozenset(hypothesis))
            return frozenset(hypothesis)


    def __missing__(self, key):
        """Return 0 mass for hypotheses that are not contained."""
        return 0.0
    
    def __copy__(self):
        c = BaseMassFunction()
        for k, v in self.items():
            c[k] = v
        return c
    
    def copy(self):
        """Creates a shallow copy of the mass function."""
        return self.__copy__()
    
    def __contains__(self, hypothesis):
        return dict.__contains__(self, BaseMassFunction._convert(hypothesis))
    
    def __getitem__(self, hypothesis):
        return dict.__getitem__(self, BaseMassFunction._convert(hypothesis))
    
    def __setitem__(self, hypothesis, value):
        """
        Adds or updates the mass value of a hypothesis.
        
        'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        In case of a negative mass value, a ValueError is raised.
        """
        #if value < 0.0:
        #    raise ValueError("mass value is negative: %f" % value)
        dict.__setitem__(self, BaseMassFunction._convert(hypothesis), value)
    
    def __delitem__(self, hypothesis):
        return dict.__delitem__(self, BaseMassFunction._convert(hypothesis))
    
    def frame(self):
        """
        Returns the frame of discernment of the mass function as a 'frozenset'.
        
        The frame of discernment is the union of all contained hypotheses.
        In case the mass function does not contain any hypotheses, an empty set is returned.
        """
        if not self:
            return frozenset()
        else:
            return frozenset.union(*self.keys()) # 就是求&

    def singletons(self):
        """
        Returns the set of all singleton hypotheses.
        
        Like 'frame()', except that each singleton is wrapped in a frozenset
        and can thus be directly passed to methods like 'bel()'.
        """
        return {frozenset((s,)) for s in self.frame()}
    
    def focal(self):
        """
        返回所有焦点假设的集合。

        焦点假设的质量值大于0。
        Returns the set of all focal hypotheses.
        
        A focal hypothesis has a mass value greater than 0.
        """
        return {h for (h, v) in self.items() if v > 0}

    def core(self, *mass_functions):
        """
        将一个或多个质量函数的核心作为“frozenset”返回。

        单一质量函数的核心是其所有焦点假设的联合。
        如果一个质量函数不包含任何焦点假设，它的核心就是一个空集。
        如果给定多个质量函数，则返回它们的组合核心(所有单个核心的交集)。
        Returns the core of one or more mass functions as a 'frozenset'.
        
        The core of a single mass function is the union of all its focal hypotheses.
        In case a mass function does not contain any focal hypotheses, its core is an empty set.
        If multiple mass functions are given, their combined core (intersection of all single cores) is returned.
        """
        if mass_functions:
            return frozenset.intersection(self.core(), *[m.core() for m in mass_functions]) #求 |
        
        else:
            focal = self.focal()
            if not focal:
                return frozenset()
            else:
                return frozenset.union(*focal)
    def __and__(self, mass_function):
        """Shorthand for 'combine_conjunctive(mass_function)'."""
        return self.combine_conjunctive(mass_function)
    
    def __or__(self, mass_function):
        """Shorthand for 'combine_disjunctive(mass_function)'."""
        return self.combine_disjunctive(mass_function)
    
    def __str__(self):
        hyp = sorted([(v, h) for (h, v) in self.items()], reverse=True)
        return "{" + "; ".join([str(set(h)) + ":" + str(v) for (v, h) in hyp]) + "}"
    
    def __mul__(self, scalar):
        if not isinstance(scalar, float):
            raise TypeError('Can only multiply by a float value.')
        m = MassFunction()
        for (h, v) in self.items():
            m[h] = v * scalar
        return m
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    def __add__(self, m):
        if not isinstance(m, BaseMassFunction):
            raise TypeError('Can only add two mass functions.')
        result = self.copy()
        for (h, v) in m.items():
            result[h] += v
        return result
    
    #重载减法__sub__
    def __sub__(self,m):
        if not isinstance(m, BaseMassFunction):
            raise TypeError('Can only sub two mass functions.')
        result = self.copy()
        for (h, v) in m.items():
            result[h] -= v
        return result

    def combine_conjunctive(self, mass_function, normalization=True, sample_count=None, importance_sampling=False):
        """
        Conjunctively combines the mass function with another mass function and returns the combination as a new mass function.
        
        The other mass function is assumed to be defined over the same frame of discernment.
        If 'mass_function' is not of type MassFunction, it is assumed to be an iterable containing multiple mass functions that are iteratively combined.
        
        If the mass functions are flatly contracting or if one of the mass functions is empty, an empty mass function is returned.
        
        'normalization' determines whether the resulting mass function is normalized (default is True).
         
        If 'sample_count' is not None, the true combination is approximated using the specified number of samples.
        In this case, 'importance_sampling' determines the method of approximation (only if normalization=True, otherwise 'importance_sampling' is ignored).
        The default method (importance_sampling=False) independently generates samples from both mass functions and computes their intersections.
        If importance_sampling=True, importance sampling is used to avoid empty intersections, which leads to a lower approximation error but is also slower.
        This method should be used if there is significant evidential conflict between the mass functions.
        """
        return self._combine(mass_function, rule=lambda s1, s2: s1 & s2, normalization=normalization, sample_count=sample_count, importance_sampling=importance_sampling)
    
    def combine_disjunctive(self, mass_function, sample_count=None):
        """
        Disjunctively combines the mass function with another mass function and returns the combination as a new mass function.
        
        The other mass function is assumed to be defined over the same frame of discernment.
        If 'mass_function' is not of type MassFunction, it is assumed to be an iterable containing multiple mass functions that are iteratively combined.
        
        If 'sample_count' is not None, the true combination is approximated using the specified number of samples.
        """
        return self._combine(mass_function, rule=lambda s1, s2: s1 | s2, normalization=False, sample_count=sample_count, importance_sampling=False)


    def _combine(self, mass_function, rule, normalization, sample_count, importance_sampling):
        """Helper method for combining two or more mass functions."""
        combined = self
        if isinstance(mass_function, BaseMassFunction):
            mass_function = [mass_function] # wrap single mass function
        for m in mass_function:
            if not isinstance(m, BaseMassFunction):
                raise TypeError("expected type MassFunction but got %s; make sure to use keyword arguments for anything other than mass functions" % type(m))
            if sample_count == None:
                combined = combined._combine_deterministic(m, rule)
            else:
                if importance_sampling and normalization:
                    combined = combined._combine_importance_sampling(m, sample_count)
                else:
                    combined = combined._combine_direct_sampling(m, rule, sample_count)
        if normalization:
            return combined.normalize()
        else:
            return combined

    def _combine_deterministic(self, mass_function, rule):
        """Helper method for deterministically combining two mass functions."""
        combined = BaseMassFunction()
        for (h1, v1) in self.items():
            for (h2, v2) in mass_function.items():
                combined[rule(h1, h2)] += v1 * v2
        return combined
    
    def _combine_direct_sampling(self, mass_function, rule, sample_count):
        """Helper method for approximatively combining two mass functions using direct sampling."""
        combined = BaseMassFunction()
        samples1 = self.sample(sample_count)
        samples2 = mass_function.sample(sample_count)
        for i in range(sample_count):
            combined[rule(samples1[i], samples2[i])] += 1.0 / sample_count
        return combined
    
    def _combine_importance_sampling(self, mass_function, sample_count):
        """Helper method for approximatively combining two mass functions using importance sampling."""
        combined = BaseMassFunction()
        for (s1, n) in self.sample(sample_count, as_dict=True).items():
            weight = mass_function.pl(s1)
            for s2 in mass_function.condition(s1).sample(n):
                combined[s2] += weight
        return combined

    def normalize(self):
        """
        Normalizes the mass function in-place.
        
        Sets the mass value of the empty set to 0 and scales all other values such that their sum equals 1.
        For convenience, the method returns 'self'.
        """
        if frozenset() in self:
            del self[frozenset()]
        mass_sum = fsum(self.values())
        if mass_sum != 1.0:
            for (h, v) in self.items():
                self[h] = v / mass_sum
        return self
    
    def conflict_log(self, mass_function, sample_count=None):
        """
        Calculates the weight of conflict between two or more mass functions.
        
        If 'mass_function' is not of type MassFunction, it is assumed to be an iterable containing multiple mass functions.
        
        The weight of conflict is computed as the (natural) logarithm of the normalization constant in Dempster's rule of combination.
        Returns infinity in case the mass functions are flatly contradicting.
        """
        # compute full conjunctive combination (could be more efficient)
        m = self.combine_conjunctive(mass_function, normalization=False, sample_count=sample_count)
        print(m)
        empty = m[frozenset()]
        m_sum = fsum(m.values())
        diff = m_sum - empty
        if diff == 0.0:
            return float('inf')
        else:
            return -log(diff)
    def conflict(self, mass_function, sample_count=None):
        m = self.combine_conjunctive(mass_function, normalization=False, sample_count=sample_count)
        empty = m[frozenset()]
        m_sum = fsum(m.values())
        diff = m_sum - empty
        if diff == 0.0:
            return float('inf')
        else:
            return empty

