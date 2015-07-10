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

import array
import operator as op

from .. import functions as fun
from .solver_registry import register_solver
from .util import Solver, _copydoc
from . import util

@register_solver('nelder-mead',
                 'simplex method for unconstrained optimization',
                 ['Simplex method for unconstrained optimization',
                 ' ',
                 'The simplex algorithm is a simple way to optimize a fairly well-behaved function.',
                 'The function is assumed to be convex. If not, this solver may yield poor solutions.',
                 ' ',
                 'This solver requires the following arguments:',
                 '- start :: starting point for the solver (through kwargs)',
                 '- ftol :: accuracy up to which to optimize the function (default 1e-4)'
                 ])
class NelderMead(Solver):
    """
    .. include:: /global.rst

    Please refer to |nelder-mead| for details about this algorithm.

    >>> s = NelderMead(x=1, y=1, xtol=1e-8) #doctest:+SKIP
    >>> best_pars, _ = s.optimize(lambda x, y: -x**2 - y**2) #doctest:+SKIP
    >>> [math.fabs(best_pars['x']) < 1e-8, math.fabs(best_pars['y']) < 1e-8]  #doctest:+SKIP
    [True, True]

    """

    def __init__(self, ftol=1e-4, max_iter=None, **kwargs):
        """Initializes the solver with a tuple indicating parameter values.

        >>> s = NelderMead(x=1, ftol=2) #doctest:+SKIP
        >>> s.start #doctest:+SKIP
        {'x': 1}
        >>> s.ftol #doctest:+SKIP
        2

        .. warning:: |warning-unconstrained|

        """

        self._start = kwargs
        self._ftol = ftol
        self._max_iter = max_iter
        if max_iter is None:
            self._max_iter = len(kwargs) * 200

    @staticmethod
    def suggest_from_seed(num_evals, **kwargs):
        """Verify that we can effectively make a solver.

        >>> s = NelderMead.suggest_from_seed(30, x=1.0, y=-1.0, z=2.0)
        >>> solver = NelderMead(**s)

        """
        return kwargs

    @property
    def ftol(self):
        """Returns the tolerance."""
        return self._ftol

    @property
    def max_iter(self):
        """Returns the maximum number of iterations."""
        return self._max_iter

    @property
    def start(self):
        """Returns the starting point."""
        return self._start

    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=True, pmap=map):
        if maximize:
            f = fun.negated(f)

        sortedkeys = sorted(self.start.keys())
        x0 = [float(self.start[k]) for k in sortedkeys]

        f = fun.static_key_order(sortedkeys)(f)

        def func(x):
            return util.score(f(*x))

        xopt = self._solve(func, x0)
        return dict([(k, v) for k, v in zip(sortedkeys, xopt)]), None

    def _solve(self, func, x0):
        def f(x):
            return func(list(x))

        x0 = array.array('f', x0)
        N = len(x0)

        vertices = [x0]
        values = [f(x0)]

        # defaults taken from Wikipedia and SciPy
        alpha = 1.; gamma = 2.; rho = -0.5; sigma = 0.5;
        nonzdelt = 0.05
        zdelt = 0.00025

        # generate vertices
        for k in range(N):
            vert = vertices[0][:]
            if vert[k] != 0:
                vert[k] = (1 + nonzdelt) * vert[k]
            else:
                vert[k] = zdelt

            vertices.append(vert)
            values.append(f(vert))

        niter = 1
        while niter < self.max_iter:

            # sort vertices by ftion value
            vertices, values = NelderMead.sort_vertices(vertices, values)

            # check for convergence
            if abs(values[0] - values[-1]) <= self.ftol:
                break

            niter += 1

            # compute center of gravity
            x0 = NelderMead.simplex_center(vertices[:-1])

            # reflect
            xr = NelderMead.reflect(x0, vertices[-1], alpha)
            fxr = f(xr)
            if values[0] < fxr < values[-2]:
                vertices[-1] = xr
                values[-1] = fxr
                continue

            # expand
            if fxr < values[0]:
                xe = NelderMead.reflect(x0, vertices[-1], gamma)
                fxe = f(xe)
                if fxe < fxr:
                    vertices[-1] = xe
                    values[-1] = fxe
                else:
                    vertices[-1] = xr
                    values[-1] = fxr
                continue

            # contract
            xc = NelderMead.reflect(x0, vertices[-1], rho)
            fxc = f(xc)
            if fxc < values[-1]:
                vertices[-1] = xc
                values[-1] = fxc
                continue

            # reduce
            for idx in range(1, len(vertices)):
                vertices[idx] = NelderMead.reflect(vertices[0], vertices[idx],
                                                   sigma)
                values[idx] = f(vertices[idx])

        return list(vertices[min(enumerate(values), key=op.itemgetter(1))[0]])

    @staticmethod
    def simplex_center(vertices):
        vector_sum = map(sum, zip(*vertices))
        return array.array('f', map(lambda x: x / len(vertices), vector_sum))

    @staticmethod
    def sort_vertices(vertices, values):
        sort_idx, values = zip(*sorted(enumerate(values), key=op.itemgetter(1)))
        # doing the same with a map bugs out, for some reason
        vertices = [vertices[x] for x in sort_idx]
        return vertices, list(values)

    @staticmethod
    def scale(vertex, coeff):
        return array.array('f', map(lambda x: coeff * x, vertex))

    @staticmethod
    def reflect(x0, xn1, alpha):
        diff = map(op.sub, x0, xn1)
        xr = array.array('f', map(op.add, x0, NelderMead.scale(diff, alpha)))
        return xr
