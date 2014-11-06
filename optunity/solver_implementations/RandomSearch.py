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

import random
import operator as op

from .. import functions as fun
from ..solver_registry import register_solver
from ..solvers import Solver, _copydoc

@register_solver('random search',
                 'random parameter tuples sampled uniformly within box constraints',
                 ['Tests random parameter tuples sampled uniformly within the box constraints.',
                  ' ',
                  'This function requires the following arguments:',
                  '- num_evals :: number of tuples to test',
                  '- box constraints via keywords: constraints are lists [lb, ub]',
                  ' ',
                  'This solver performs num_evals function evaluations.',
                  ' ',
                  'This solver implements the technique described here:',
                  'Bergstra, James, and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of Machine Learning Research 13 (2012): 281-305.']
                 )
class RandomSearch(Solver):

    def __init__(self, num_evals, **kwargs):
        """Initializes the solver with bounds and a number of allowed evaluations.
        kwargs must be a dictionary of parameter-bound pairs representing the box constraints.
        Bounds are a 2-element list: [lower_bound, upper_bound].

        >>> s = RandomSearch(x=[0, 1], y=[-1, 2], num_evals=50)
        >>> s.bounds
        {'y': [-1, 2], 'x': [0, 1]}
        >>> s.num_evals
        50

        """
        assert all([len(v) == 2 and v[0] <= v[1]
                    for v in kwargs.values()]), 'kwargs.values() are not [lb, ub] pairs'
        self._bounds = kwargs
        self._num_evals = num_evals

    @staticmethod
    def suggest_from_box(num_evals, **kwargs):
        d = dict(kwargs)
        d['num_evals'] = num_evals
        return d

    @property
    def upper(self, par):
        """Returns the upper bound of par."""
        return self._bounds[par][1]

    @property
    def lower(self, par):
        """Returns the lower bound of par."""
        return self._bounds[par][0]

    @property
    def bounds(self):
        """Returns a dictionary containing the box constraints."""
        return self._bounds

    @property
    def num_evals(self):
        """Returns the number of evaluations this solver may do."""
        return self._num_evals

    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=True, pmap=map):

        def generate_rand_args(len=1):
            return [[random.uniform(bounds[0], bounds[1]) for _ in range(len)]
                    for _, bounds in sorted(self.bounds.items())]

        best_pars = None
        sortedkeys = sorted(self.bounds.keys())
        f = fun.static_key_order(sortedkeys)(f)

        if maximize:
            comp = lambda score, best: score > best
        else:
            comp = lambda score, best: score < best

        tuples = generate_rand_args(self.num_evals)
        scores = pmap(f, *tuples)

        if maximize:
            comp = max
        else:
            comp = min
        best_idx, _ = comp(enumerate(scores), key=op.itemgetter(1))
        best_pars = op.itemgetter(best_idx)(zip(*tuples))
        return dict([(k, v) for k, v in zip(sortedkeys, best_pars)]), None
