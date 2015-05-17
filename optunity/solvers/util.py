#! /usr/bin/env python

# Copyright (c) 2014 KU Leuven, ESAT-STADIUS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither name of copyright holders nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import abc
import random
import threading

def uniform_in_bounds(bounds):
    """Generates a random uniform sample between ``bounds``.

    :param bounds: the bounds we must adhere to
    :type bounds: dict {"name": [lb ub], ...}
    """
    return map(random.uniform, *zip(*bounds.values()))

def scale_unit_to_bounds(seq, bounds):
    """
    Scales all elements in seq (unit hypercube) to the box constraints in bounds.

    :param seq: the sequence in the unit hypercube to scale
    :type seq: iterable
    :param bounds: bounds to scale to
    :type seq: iterable of [lb, ub] pairs
    :returns: a list of scaled elements of `seq`

    >>> scale_unit_to_bounds([0.0, 0.5, 0.5, 1.0], [[-1.0, 2.0], [-2.0, 0.0], [0.0, 3.0], [0.0, 2.0]])
    [-1.0, -1.0, 1.5, 2.0]

    """
    assert(len(seq) == len(bounds))
    return [float(x) * float(b[1] - b[0]) + b[0]
            for x, b in zip(seq, bounds)]


# python version-independent metaclass usage
SolverBase = abc.ABCMeta('SolverBase', (object, ), {})

class Solver(SolverBase):
    """Base class of all Optunity solvers.
    """

    @abc.abstractmethod
    def optimize(self, f, maximize=True, pmap=map):
        """Optimizes ``f``.

        :param f: the objective function
        :type f: callable
        :param maximize: do we want to maximizes?
        :type maximize: boolean
        :param pmap: the map() function to use
        :type pmap: callable
        :returns:
            - the arguments which optimize ``f``
            - an optional solver report, can be None

        """
        pass

    def maximize(self, f, pmap=map):
        """Maximizes f.

        :param f: the objective function
        :type f: callable
        :param pmap: the map() function to use
        :type pmap: callable
        :returns:
            - the arguments which optimize ``f``
            - an optional solver report, can be None

        """
        return self.optimize(f, True, pmap=pmap)

    def minimize(self, f, pmap=map):
        """Minimizes ``f``.

        :param f: the objective function
        :type f: callable
        :param pmap: the map() function to use
        :type pmap: callable
        :returns:
            - the arguments which optimize ``f``
            - an optional solver report, can be None

        """
        return self.optimize(f, False, pmap=pmap)


# http://stackoverflow.com/a/13743316
def _copydoc(fromfunc, sep="\n"):
    """
    Decorator: Copy the docstring of `fromfunc`
    """
    def _decorator(func):
        sourcedoc = fromfunc.__doc__
        if func.__doc__ == None:
            func.__doc__ = sourcedoc
        else:
            func.__doc__ = sep.join([sourcedoc, func.__doc__])
        return func
    return _decorator

def shrink_bounds(bounds, coverage=0.99):
    """Shrinks the bounds. The new bounds will cover the fraction ``coverage``.

    >>> [round(x, 3) for x in shrink_bounds([0, 1], coverage=0.99)]
    [0.005, 0.995]

    """
    def shrink(lb, ub, coverage):
        new_range = float(ub - lb) * coverage / 2
        middle = float(ub + lb) / 2
        return [middle-new_range, middle+new_range]

    return dict([(k, shrink(v[0], v[1], coverage))
                 for k, v in bounds.items()])


def score(value):
    """General wrapper around objective function evaluations to get the score.

    :param value: output of the objective function
    :returns: the score

    If value is a scalar, it is returned immediately. If value is iterable, its first element is returned.
    """
    try:
        return value[0]
    except (TypeError, IndexError):
        return value


class ThreadSafeQueue(object):
    def __init__(self, lst=None):
        """
        Initializes a new object.

        :param lst: initial content
        :type lst: list or None
        """
        if lst: self._content = lst
        else: self._content = []
        self._lock = threading.Lock()

    @property
    def lock(self): return self._lock

    @property
    def content(self): return self._content

    def append(self, value):
        """
        Acquires lock and appends value to the content.

        >>> q1 = ThreadSafeQueue()
        >>> q1
        []
        >>> q1.append(1)
        [1]

        """
        with self.lock: self.content.append(value)

    def __iter__(self):
        for i in self.content:
            yield i

    def __len__(self): return len(self.content)
    def __getitem__(self, idx): return self.content[idx]
    def __repr__(self): return str(self.content)

    def copy(self):
        """
        Makes a deep copy of this ThreadSafeQueue.

        >>> q1 = ThreadSafeQueue([1,2,3])
        >>> q2 = q1.copy()
        >>> q2.append(4)
        >>> q1
        [1, 2, 3]
        >>> q2
        [1, 2, 3, 4]

        """
        return ThreadSafeQueue(self.content[:])

