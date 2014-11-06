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

from .. import functions as fun
from ..solver_registry import register_solver
from ..solvers import Solver, _copydoc

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
    Exhaustive search over the Cartesian product of parameter tuples.
    Returns x (the tuple which maximizes f) and its score f(x).

    >>> s = GridSearch(x=[1,2,3], y=[-1,0,1])
    >>> best_pars, _ = s.optimize(lambda x, y: x*y)
    >>> best_pars
    {'y': 1, 'x': 3}

    """

    def __init__(self, **kwargs):
        """Initializes the solver with a tuple indicating parameter values.

        >>> s = GridSearch(x=[1,2], y=[3,4])
        >>> s.parameter_tuples
        {'y': [3, 4], 'x': [1, 2]}

        """
        self._parameter_tuples = kwargs

    @staticmethod
    def suggest_from_box(num_evals, **kwargs):
        return kwargs

    @property
    def parameter_tuples(self):
        """Returns the possible values of every parameter."""
        return self._parameter_tuples

    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=True, pmap=map):

        best_pars = None
        sortedkeys = sorted(self.parameter_tuples.keys())
        f = fun.static_key_order(sortedkeys)(f)

        if maximize:
            comp = lambda score, best: score > best
        else:
            comp = lambda score, best: score < best

        tuples = list(zip(*itertools.product(*zip(*sorted(self.parameter_tuples.items()))[1])))
        scores = pmap(f, *tuples)

        if maximize:
            comp = max
        else:
            comp = min
        best_idx, _ = comp(enumerate(scores), key=op.itemgetter(1))
        best_pars = op.itemgetter(best_idx)(zip(*tuples))
        return dict([(k, v) for k, v in zip(sortedkeys, best_pars)]), None
