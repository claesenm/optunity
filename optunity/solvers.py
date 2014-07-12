#! /usr/bin/env python

# Author: Marc Claesen
#
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

import itertools
import collections
import abc
import functools
import random
import operator

# optunity imports
from . import functions as fun
from .solver_registry import register_solver

# python version-independent metaclass usage
SolverBase = abc.ABCMeta('SolverBase', (object, ), {})


class Solver(SolverBase):
    """A callable which maximizes its argument (also a callable).
    """
    @abc.abstractmethod
    def maximize(self, f):
        """Maximizes f. Returns the optimal arguments and a
        solver report (can be None).
        """
        pass


@register_solver('grid-search',
                 'finds optimal parameter values through grid search',
                 ['TODO'])
class GridSearch(Solver):

    def __init__(self, **kwargs):
        """Initializes the solver with a tuple indicating parameter values.

        >>> s = GridSearch(x=[1,2], y=[3,4])
        >>> s.parameter_tuples
        {'y': [3, 4], 'x': [1, 2]}

        """
        self._parameter_tuples = kwargs

    @property
    def parameter_tuples(self):
        """Returns the possible values of every parameter."""
        return self._parameter_tuples

    def maximize(self, f):
        """
        Exhaustive search over the Cartesian product of parameter tuples.
        Returns x (the tuple which maximizes f) and its score f(x).

        >>> s = GridSearch(x=[1,2,3], y=[-1,0,1])
        >>> best_pars, _ = s.maximize(lambda x, y: x*y)
        >>> best_pars
        {'y': 1, 'x': 3}

        """
        best_score = float("-inf")
        best_pars = None

        sortedkeys = sorted(self.parameter_tuples.keys())
        f = fun.static_key_order(sortedkeys)(f)

        for pars in itertools.product(*zip(*sorted(self.parameter_tuples.items()))[1]):
            score = f(pars)
            if score > best_score:
                best_score = score
                best_pars = pars

        # no useful statistics to report
        return dict([(k, v) for k, v in zip(sortedkeys, best_pars)]), None


@register_solver('random-search',
                 'finds optimal parameter values through random search',
                 ['This solver implements the technique described here:',
                  'Bergstra, James, and Yoshua Bengio. Random search for hyper-parameter optimization. The Journal of Machine Learning Research 13 (2012): 281-305.']
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

    def maximize(self, f):
        """
        TODO
        """

        def generate_rand_args():
            return dict([(par, random.uniform(bounds[0], bounds[1]))
                         for par, bounds in self.bounds.items()])

        parameter_tuples = [generate_rand_args()
                            for _ in range(self.num_evals)]
        best_score = float("-inf")
        best_pars = None

        for pars in parameter_tuples:
            score = f(**pars)
            if score > best_score:
                best_score = score
                best_pars = pars

        return best_pars, None  # no useful statistics to report


@register_solver('direct',
                 'optimizes a box-constrained function using the DIRECT algorithm',
                 ['This solver implements the technique described here:',
                  'Jones, Donald R., Cary D. Perttunen, and Bruce E. Stuckman. "Lipschitzian optimization without the Lipschitz constant." Journal of Optimization Theory and Applications 79.1 (1993): 157-181.']
                 )
class Direct(Solver):

    def __init__(self, num_evals, eps, **kwargs):
        """Initializes the solver with bounds and a number of allowed evaluations.
        kwargs must be a dictionary of parameter-bound pairs representing the box constraints.
        Bounds are a 2-element list: [lower_bound, upper_bound].

        >>> s = Direct(x=[0, 1], y=[-1, 2], num_evals=50)
        >>> s.bounds
        {'y': [-1, 2], 'x': [0, 1]}
        >>> s.num_evals
        50

        """
        assert all([len(v) == 2 and v[0] <= v[1]
                    for v in kwargs.values()]), 'kwargs.values() are not [lb, ub] pairs'
        tup = collections.namedtuple('args',kwargs.keys())
        self._lower = tup(**dict([(k, v[0]) for k, v in kwargs.items()]))
        self._upper = tup(**dict([(k, v[1]) for k, v in kwargs.items()]))
        self._num_evals = num_evals
        self._eps = eps

    def scale_data(self, data):
        return dict([(k, lb + float(x) * (ub - lb))
                     for k, x, lb, ub in zip(self.lower._fields, data,
                                             self.lower, self.upper)])

    @property
    def tup(self):
        return self._tup

    @property
    def eps(self):
        return self._eps

    @property
    def upper(self, par=None):
        """Returns the upper bound of par.

        If par is None, returns all upper bounds."""
        if par is None:
            return self._upper
        return getattr(self._upper, par)

    @property
    def lower(self, par=None):
        """Returns the lower bound of par.

        If par is None, returns all lower bounds."""
        if par is None:
            return self._lower
        return getattr(self._lower, par)

    @property
    def bounds(self):
        """Returns a dictionary containing the box constraints."""
        return dict([(k, [lb, ub])
                     for k, lb, ub in zip(self.lower._fields,
                                          self.lower, self.upper)])

    @property
    def num_evals(self):
        """Returns the number of evaluations this solver may do."""
        return self._num_evals

    def maximize(self, f):
        """
        TODO
        """
        def generate_rand_args():
            return dict([(par, random.uniform(bounds[0], bounds[1]))
                         for par, bounds in self.bounds.items()])

        parameter_tuples = [generate_rand_args()
                            for _ in range(self.num_evals)]
        best_score = float("-inf")
        best_pars = None

        for pars in parameter_tuples:
            score = f(**pars)
            if score > best_score:
                best_score = score
                best_pars = pars

        return best_pars, None  # no useful statistics to report

try:
    import scipy.optimize
    import numpy as np

    @register_solver('nelder-mead',
                     'optimizes parameters using the downhill simplex method',
                     ['TODO'])
    class NelderMead(Solver):

        def __init__(self, x0, xtol):
            """Initializes the solver with a tuple indicating parameter values.

            >>> s = NelderMead(x0={'x': 1}, xtol=2)
            >>> s.x0
            {'x': 1}
            >>> s.xtol
            2

            """
            self._x0 = x0
            self._xtol = xtol

        @property
        def xtol(self):
            """Returns the tolerance."""
            return self._xtol

        @property
        def x0(self):
            """Returns the starting point x0."""
            return self._x0

        def maximize(self, f):
            """
            Performs Nelder-Mead optimization to minimize f. Requires scipy.

            In scipy < 0.11.0, scipy.optimize.fmin is used.
            In scipy >= 0.11.0, scipy.optimize.minimize is used.

            >>> s = NelderMead({'x': 1, 'y': 1}, 1e-8)
            >>> best_pars, _ = s.maximize(lambda x, y: -x**2 - y**2)
            >>> [math.fabs(best_pars['x']) < 1e-8, math.fabs(best_pars['y']) < 1e-8]
            [True, True]

            """
            # Nelder-Mead implicitly minimizes, so negate f()
            f = fun.negated(f)

            sortedkeys = sorted(self.x0.keys())
            x0 = [self.x0[k] for k in sortedkeys]
            f = fun.static_key_order(sortedkeys)(f)

            version = scipy.__version__
            if int(version.split('.')[1]) >= 11:
                print('HALP: wrong scipy version')
                pass  # TODO
            else:
                xopt = scipy.optimize.fmin(f, np.array(x0),
                                           xtol=self.xtol, disp=False)
                return dict([(k, v) for k, v in zip(sortedkeys, xopt)]), None

except ImportError:
    pass

try:
    import deap
    import deap.creator
    import deap.base
    import deap.tools

# http://deap.gel.ulaval.ca/doc/dev/examples/pso_basic.html
# https://code.google.com/p/deap/source/browse/examples/pso/basic.py?name=dev
    @register_solver('particle-swarm',
                     'particle swarm optimization',
                     ['TODO'])
    class ParticleSwarm(Solver):

        # TODO: implement warm start
        def __init__(self, num_particles, num_generations,
                     max_speed, **kwargs):
            """blah"""
            assert all([len(v) == 2 and v[0] <= v[1]
                        for v in kwargs.values()]), 'kwargs.values() are not [lb, ub] pairs'
            self._bounds = kwargs
            self._ttype = collections.namedtuple('ttype', kwargs.keys())
            self._num_particles = num_particles
            self._num_generations = num_generations
            self._max_speed = max_speed
            self._smax = [self.max_speed * (b[1] - b[0])
                          for _, b in self.bounds.items()]
            self._smin = map(operator.neg, self.smax)

            deap.creator.create("FitnessMax", deap.base.Fitness,
                                weights=(1.0,))
            deap.creator.create("Particle", list,
                                fitness=deap.creator.FitnessMax, speed=list,
                                best=None)
            self._toolbox = deap.base.Toolbox()
            self._toolbox.register("particle", self.generate)
            self._toolbox.register("population", deap.tools.initRepeat, list,
                                   self.toolbox.particle)
            self._toolbox.register("update", self.updateParticle,
                                   phi1=2.0, phi2=2.0)

        @property
        def num_particles(self):
            return self._num_particles

        @property
        def num_generations(self):
            return self._num_generations

        @property
        def toolbox(self):
            return self._toolbox

        @property
        def max_speed(self):
            return self._max_speed

        @property
        def smax(self):
            return self._smax

        @property
        def smin(self):
            return self._smin

        @property
        def bounds(self):
            return self._bounds

        @property
        def ttype(self):
            return self._ttype

        def generate(self):
            part = deap.creator.Particle(random.uniform(bounds[0], bounds[1])
                                         for _, bounds in self.bounds.items())
            part.speed = [random.uniform(smin, smax)
                          for smin, smax in zip(self.smin, self.smax)]
            return part

        def updateParticle(self, part, best, phi1, phi2):
            u1 = (random.uniform(0, phi1) for _ in range(len(part)))
            u2 = (random.uniform(0, phi2) for _ in range(len(part)))
            v_u1 = map(operator.mul, u1,
                       map(operator.sub, part.best, part))
            v_u2 = map(operator.mul, u2,
                       map(operator.sub, best, part))
            part.speed = list(map(operator.add, part.speed,
                                  map(operator.add, v_u1, v_u2)))
            for i, speed in enumerate(part.speed):
                if speed < self.smin[i]:
                    part.speed[i] = self.smin[i]
                elif speed > self.smax[i]:
                    part.speed[i] = self.smax[i]
            part[:] = list(map(operator.add, part, part.speed))

        def maximize(self, f):
            """
            TODO
            """
            def evaluate(individual):
                return (f(**dict([(k, v)
                                  for k, v in zip(self.bounds.keys(),
                                                  individual)])),)
            self._toolbox.register("evaluate", evaluate)

            pop = self.toolbox.population(self.num_particles)
            best = None

            for g in range(self.num_generations):
                for part in pop:
                    part.fitness.values = self.toolbox.evaluate(part)
                    if not part.best or part.best.fitness < part.fitness:
                        part.best = deap.creator.Particle(part)
                        part.best.fitness.values = part.fitness.values
                    if not best or best.fitness < part.fitness:
                        best = deap.creator.Particle(part)
                        best.fitness.values = part.fitness.values
                for part in pop:
                    self.toolbox.update(part, best)

            return dict([(k, v)
                         for k, v in zip(self.bounds.keys(), best)]), None

except ImportError:
    pass

