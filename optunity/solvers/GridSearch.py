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

import operator as op
import itertools

from ..functions import static_key_order
from .solver_registry import register_solver
from .util import Solver, _copydoc, shrink_bounds
from . import util

# http://stackoverflow.com/a/15978862
def nth_root(val, n):
    ret = int(val**(1./n))
    return ret + 1 if (ret + 1) ** n == val else ret

@register_solver('grid search',
                 'finds optimal parameter values on a predefined grid',
                 ['Retrieves the best parameter tuple on a predefined grid.',
                  ' ',
                  'This function requires the grid to be specified via named arguments:',
                  '- names :: argument names',
                  '- values :: list of grid coordinates to test',
                  ' ',
                  'The solver performs evaluation on the Cartesian product of grid values.',
                  'The number of evaluations is the product of the length of all value vectors.'
                  ])
class GridSearch(Solver):
    """
    .. include:: /global.rst

    Please refer to |gridsearch| for more information about this algorithm.

    Exhaustive search over the Cartesian product of parameter tuples.
    Returns x (the tuple which maximizes f) and its score f(x).

    >>> s = GridSearch(x=[1,2,3], y=[-1,0,1])
    >>> best_pars, _ = s.optimize(lambda x, y: x*y)
    >>> best_pars['x']
    3
    >>> best_pars['y']
    1

    """

    def __init__(self, **kwargs):
        """Initializes the solver with a tuple indicating parameter values.

        >>> s = GridSearch(x=[1,2], y=[3,4])
        >>> s.parameter_tuples['x']
        [1, 2]
        >>> s.parameter_tuples['y']
        [3, 4]

        """
        self._parameter_tuples = kwargs

    @staticmethod
    def assign_grid_points(lb, ub, density):
        """Assigns equally spaced grid points with given density in [ub, lb].
        The bounds are always used. ``density`` must be at least 2.

        :param lb: lower bound of resulting grid
        :param ub: upper bound of resulting grid
        :param density: number of points to use
        :type lb: float
        :type ub: float
        :type density: int

        >>> s = GridSearch.assign_grid_points(1.0, 2.0, 3)
        >>> s #doctest:+SKIP
        [1.0, 1.5, 2.0]

        """
        density = int(density)
        assert(density >= 2)
        step = float(ub-lb)/(density-1)
        return [lb+i*step for i in range(density)]

    @staticmethod
    def suggest_from_box(num_evals, **kwargs):
        """Creates a GridSearch solver that uses less than num_evals evaluations
        within given bounds (lb, ub). The bounds are first tightened, resulting in
        new bounds covering 99% of the area.

        The resulting solver will use an equally spaced grid with the same number
        of points in every dimension. The amount of points that is used is per
        dimension is the nth root of num_evals, rounded down, where n is the number
        of hyperparameters.

        >>> s = GridSearch.suggest_from_box(30, x=[0, 1], y=[-1, 0], z=[-1, 1])
        >>> s['x'] #doctest:+SKIP
        [0.005, 0.5, 0.995]
        >>> s['y'] #doctest:+SKIP
        [-0.995, -0.5, -0.005]
        >>> s['z'] #doctest:+SKIP
        [-0.99, 0.0, 0.99]


        Verify that we can effectively make a solver from box.

        >>> s = GridSearch.suggest_from_box(30, x=[0, 1], y=[-1, 0], z=[-1, 1])
        >>> solver = GridSearch(**s)

        """
        bounds = shrink_bounds(kwargs)
        num_pars = len(bounds)

        # number of grid points in each dimension
        # so we get density^num_par grid points in total
        density = nth_root(num_evals, num_pars)
        grid = dict([(k, GridSearch.assign_grid_points(b[0], b[1], density))
                     for k, b in bounds.items()])
        return grid

    @property
    def parameter_tuples(self):
        """Returns the possible values of every parameter."""
        return self._parameter_tuples

    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=True, pmap=map):

        best_pars = None
        f = static_key_order(self.parameter_tuples.keys())(f)

        if maximize:
            comp = lambda score, best: score > best
        else:
            comp = lambda score, best: score < best

        tuples = list(zip(*itertools.product(*list(zip(*self.parameter_tuples.items()))[1])))
        scores = pmap(f, *tuples)
        scores = map(util.score, scores)

        if maximize:
            comp = max
        else:
            comp = min
        best_idx, _ = comp(enumerate(scores), key=op.itemgetter(1))
        best_pars = op.itemgetter(best_idx)(list(zip(*tuples)))
        return dict([(k, v) for k, v in zip(self.parameter_tuples.keys(), best_pars)]), None
