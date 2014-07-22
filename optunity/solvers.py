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

"""Module to take care of registering solvers for use in the main Optunity API.

Main classes in this module:

* :class:`Solver`
* :class:`GridSearch`
* :class:`RandomSearch`
* :class:`DIRECT`
* :class:`NelderMead`
* :class:`ParticleSwarm`
* :class:`CMA_ES`

.. warning::
    :class:`NelderMead` is only available if SciPy_ is available.
    :class:`ParticleSwarm` and :class:`CMA_ES` require DEAP_.

    .. _SciPy: http://http://www.scipy.org/
    .. _DEAP: https://code.google.com/p/deap/


Bibliographic references for some solvers:

.. [HANSEN2001] Nikolaus Hansen and Andreas Ostermeier. *Completely
    derandomized self-adaptation in evolution  strategies*.
    Evolutionary computation, 9(2):159-195, 2001.

.. [DEAP2012] Felix-Antoine Fortin, Francois-Michel De Rainville, Marc-Andre Gardner,
    Marc Parizeau and Christian Gagne, *DEAP: Evolutionary Algorithms Made Easy*,
    Journal of Machine Learning Research, pp. 2171-2175, no 13, jul 2012.


.. moduleauthor:: Marc Claesen

"""

import itertools
import collections
import abc
import functools
import random
import operator
import math

# optunity imports
from . import functions as fun
from .solver_registry import register_solver

_scipy_available = True
try:
    import scipy.optimize
except ImportError:
    _scipy_available = False

_numpy_available = True
try:
    import numpy as np
except ImportError:
    _numpy_available = False

_deap_available = True
try:
    import deap
    import deap.creator
    import deap.base
    import deap.tools
    import deap.cma
    import deap.algorithms
except ImportError:
    _deap_available = False
except TypeError:
    # this can happen because DEAP is in Python 2
    # install needs to take proper care of converting
    # 2 to 3 when necessary
    _deap_available = False


# python version-independent metaclass usage
SolverBase = abc.ABCMeta('SolverBase', (object, ), {})

class Solver(SolverBase):
    """Base class of all Optunity solvers.
    """

    @abc.abstractmethod
    def optimize(self, f, maximize=True, pmap=map):
        """Optimizes ``f``.

        :param f: the objective function
        :param maximize: bool to indicate maximization
        :param parallelize: turn on parallelization of evaluations
        :returns:
            - the arguments which optimize ``f``
            - an optional solver report, can be None

        """
        pass

    def maximize(self, f, pmap=map):
        """Maximizes f.

        :param f: the objective function
        :param maximize: bool to indicate maximization
        :param parallelize: turn on parallelization of evaluations
        :returns:
            - the arguments which optimize ``f``
            - an optional solver report, can be None

        """
        return optimize(f, True, pmap=pmap)

    def minimize(self, f, pmap=map):
        """Minimizes ``f``.

        :param f: the objective function
        :param maximize: bool to indicate maximization
        :param parallelize: turn on parallelization of evaluations
        :returns:
            - the arguments which optimize ``f``
            - an optional solver report, can be None

        """
        return optimize(f, False, pmap=pmap)



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
    def from_box(num_evals, **kwargs):
        pass

    @property
    def parameter_tuples(self):
        """Returns the possible values of every parameter."""
        return self._parameter_tuples

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
        best_idx, _ = comp(enumerate(scores), key=operator.itemgetter(1))
        best_pars = operator.itemgetter(best_idx)(zip(*tuples))
        return dict([(k, v) for k, v in zip(sortedkeys, best_pars)]), None


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
        best_idx, _ = comp(enumerate(scores), key=operator.itemgetter(1))
        best_pars = operator.itemgetter(best_idx)(zip(*tuples))
        return dict([(k, v) for k, v in zip(sortedkeys, best_pars)]), None


@register_solver('direct',
                 'DIviding RECTangles refinement search strategy',
                 ['This solver implements the technique described here:',
                  'Jones, Donald R., Cary D. Perttunen, and Bruce E. Stuckman. Lipschitzian optimization without the Lipschitz constant. Journal of Optimization Theory and Applications 79.1 (1993): 157-181.']
                 )
class Direct(Solver):

    def __init__(self, num_evals, eps, **kwargs):
        """Initializes the solver with bounds and a number of allowed evaluations.
        kwargs must be a dictionary of parameter-bound pairs representing the box constraints.
        Bounds are a 2-element list: [lower_bound, upper_bound].

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

    def optimize(self, f, maximize=True, pmap=map):

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

class NelderMead(Solver):
    """
    Performs Nelder-Mead optimization to minimize f. Requires scipy.

    In scipy < 0.11.0, scipy.optimize.fmin is used.
    In scipy >= 0.11.0, scipy.optimize.minimize is used.

    >>> s = NelderMead(x=1, y=1, xtol=1e-8) #doctest:+SKIP
    >>> best_pars, _ = s.optimize(lambda x, y: -x**2 - y**2) #doctest:+SKIP
    >>> [math.fabs(best_pars['x']) < 1e-8, math.fabs(best_pars['y']) < 1e-8]  #doctest:+SKIP
    [True, True]

    """

    def __init__(self, xtol=1e-4, **kwargs):
        """Initializes the solver with a tuple indicating parameter values.

        >>> s = NelderMead(x=1, xtol=2) #doctest:+SKIP
        >>> s.start #doctest:+SKIP
        {'x': 1}
        >>> s.xtol #doctest:+SKIP
        2

        """
        if not _scipy_available:
            raise ImportError('This solver requires SciPy but it is missing.')

        self._start = kwargs
        self._xtol = xtol

    @staticmethod
    def suggest_from_seed(num_evals, **kwargs):
        return kwargs

    @property
    def xtol(self):
        """Returns the tolerance."""
        return self._xtol

    @property
    def start(self):
        """Returns the starting point."""
        return self._start

    def optimize(self, f, maximize=True, pmap=map):
        if maximize:
            f = fun.negated(f)

        sortedkeys = sorted(self.start.keys())
        x0 = [self.start[k] for k in sortedkeys]

        f = fun.static_key_order(sortedkeys)(f)
        def func(x):
            return f(*list(x))

        version = scipy.__version__
        if int(version.split('.')[1]) >= 11:
            print('HALP: wrong scipy version')
            pass  # TODO
        else:
            xopt = scipy.optimize.fmin(func, np.array(x0),
                                        xtol=self.xtol, disp=False)
            return dict([(k, v) for k, v in zip(sortedkeys, xopt)]), None

if _scipy_available:
    NelderMead = register_solver('nelder-mead',
                                 'simplex method for unconstrained optimization',
                                 ['Simplex method for unconstrained optimization',
                                  ' ',
                                  'The simplex algorithm is a simple way to optimize a fairly well-behaved function.',
                                  'The function is assumed to be convex. If not, this solver may yield poor solutions.',
                                  ' ',
                                  'This solver requires the following arguments:',
                                  '- start :: starting point for the solver (through kwargs)',
                                  '- xtol :: accuracy up to which to optimize the function (default 1e-4)'
                                 ])(NelderMead)


class CMA_ES(Solver):
    """
    Covariance Matrix Adaptation Evolutionary Strategy

    This solver implements the technique described in [HANSEN2001]_.
    This solver uses an implementation available in the DEAP library [DEAP2012]_.

    """

    def __init__(self, num_generations, sigma=1.0, Lambda=None, **kwargs):
        """blah"""
        if not _deap_available:
            raise ImportError('This solver requires DEAP but it is missing.')
        if not _numpy_available:
            raise ImportError('This solver requires NumPy but it is missing.')

        self._num_generations = num_generations
        self._start = kwargs
        self._sigma = sigma
        self._lambda = Lambda

    @staticmethod
    def suggest_from_seed(num_evals, **kwargs):
        fertility = 4 + 3 * math.log(len(kwargs))
        d = dict(kwargs)
        d['num_generations'] = int(math.ceil(float(num_evals) / fertility))
        # num_gen is overestimated
        # this will require slightly more function evaluations than permitted by num_evals
        return d

    @property
    def num_generations(self):
        return self._num_generations

    @property
    def start(self):
        """Returns the starting point for CMA-ES."""
        return self._start

    @property
    def lambda_(self):
        return self._lambda

    @property
    def sigma(self):
        return self._sigma

    def optimize(self, f, maximize=True, pmap=map):
        toolbox = deap.base.Toolbox()
        if maximize:
            fit = 1.0
        else:
            fit = -1.0
        deap.creator.create("FitnessMax", deap.base.Fitness,
                            weights=(fit,))
        Fit = deap.creator.FitnessMax
        deap.creator.create("Individual", list,
                            fitness=Fit)
        Individual = deap.creator.Individual

        if self.lambda_:
            strategy = deap.cma.Strategy(centroid=self.start.values(),
                                            sigma=self.sigma, lambda_=self.lambda_)
        else:
            strategy = deap.cma.Strategy(centroid=self.start.values(),
                                            sigma=self.sigma)
        toolbox.register("generate", strategy.generate, Individual)
        toolbox.register("update", strategy.update)

        @functools.wraps(f)
        def evaluate(individual):
            return (f(**dict([(k, v)
                                for k, v in zip(self.start.keys(),
                                                individual)])),)
        toolbox.register("evaluate", evaluate)
        toolbox.register("map", pmap)

        hof = deap.tools.HallOfFame(1)
        deap.algorithms.eaGenerateUpdate(toolbox=toolbox,
                                            ngen=self._num_generations,
                                            halloffame=hof, verbose=False)

        return dict([(k, v)
                        for k, v in zip(self.start.keys(), hof[0])]), None

# CMA_ES solver requires deap > 1.0.1
# http://deap.readthedocs.org/en/latest/examples/cmaes.html
if _deap_available and _numpy_available:
    CMA_ES =register_solver('cma-es', 'covariance matrix adaptation evolutionary strategy',
                        ['CMA-ES: covariance matrix adaptation evolutionary strategy',
                        ' ',
                        'This method requires the following parameters:',
                        '- num_generations :: number of generations to use',
                        '- sigma :: (optional) initial covariance, default 1',
                        '- Lambda :: (optional) measure of reproducibility',
                        '- starting point: through kwargs'
                        ' ',
                        'This method is described in detail in:',
                        'Hansen and Ostermeier, 2001. Completely Derandomized Self-Adaptation in Evolution Strategies. Evolutionary Computation'
                         ])(CMA_ES)


class ParticleSwarm(Solver):
    """
    This solver uses an implementation available in the DEAP library [DEAP2012]_.

    """

    def __init__(self, num_particles, num_generations, max_speed=None, **kwargs):
        """blah"""
        if not _deap_available:
            raise ImportError('This solver requires DEAP but it is missing.')

        assert all([len(v) == 2 and v[0] <= v[1]
                    for v in kwargs.values()]), 'kwargs.values() are not [lb, ub] pairs'
        self._bounds = kwargs
        self._ttype = collections.namedtuple('ttype', kwargs.keys())
        self._num_particles = num_particles
        self._num_generations = num_generations

        if max_speed is None:
            max_speed = 2.0/num_generations
        self._max_speed = max_speed
        self._smax = [self.max_speed * (b[1] - b[0])
                        for _, b in self.bounds.items()]
        self._smin = map(operator.neg, self.smax)

        self._toolbox = deap.base.Toolbox()
        self._toolbox.register("particle", self.generate)
        self._toolbox.register("population", deap.tools.initRepeat, list,
                                self.toolbox.particle)
        self._toolbox.register("update", self.updateParticle,
                                phi1=2.0, phi2=2.0)

    @staticmethod
    def suggest_from_box(num_evals, **kwargs):
        d = dict(kwargs)
        if num_evals > 200:
            d['num_particles'] = 50
            d['num_generations'] = int(math.ceil(float(num_evals) / 50))
        elif num_evals > 10:
            d['num_particles'] = 10
            d['num_generations'] = int(math.ceil(float(num_evals) / 10))
        else:
            d['num_particles'] = num_evals
            d['num_generations'] = 1
        return d

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
        part = self._Particle(random.uniform(bounds[0], bounds[1])
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

    def optimize(self, f, maximize=True, pmap=map):

        @functools.wraps(f)
        def evaluate(individual):
            return (f(**dict([(k, v)
                              for k, v in zip(self.bounds.keys(),
                                              individual)])),)

        self._toolbox.register("evaluate", evaluate)
        self._toolbox.register("map", pmap)

        if maximize:
            fit = 1.0
        else:
            fit = -1.0
        deap.creator.create("FitnessMax", deap.base.Fitness,
                            weights=(fit,))
        FitnessMax = deap.creator.FitnessMax

        deap.creator.create("Particle", list,
                            fitness=FitnessMax, speed=list,
                            best=None)
        self._Particle = deap.creator.Particle

        pop = self.toolbox.population(self.num_particles)
        best = None

        for g in range(self.num_generations):
            fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
            for part, fitness in zip(pop, fitnesses):
                part.fitness.values = fitness
                if not part.best or part.best.fitness < part.fitness:
                    part.best = self._Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = self._Particle(part)
                    best.fitness.values = part.fitness.values
            for part in pop:
                self.toolbox.update(part, best)

        return dict([(k, v)
                        for k, v in zip(self.bounds.keys(), best)]), None

# PSO solver requires deap > 0.7
# http://deap.gel.ulaval.ca/doc/dev/examples/pso_basic.html
# https://code.google.com/p/deap/source/browse/examples/pso/basic.py?name=dev
if _deap_available:
    ParticleSwarm = register_solver('particle swarm',
                         'particle swarm optimization',
                        ['Maximizes the function using particle swarm optimization.',
                        ' ',
                        'This is a two-phase approach:',
                        '1. Initialization: randomly initializes num_particles particles.',
                        '   Particles are randomized uniformly within the box constraints.',
                        '2. Iteration: particles move during num_generations iterations.',
                        '   Movement is based on their velocities and mutual attractions.',
                       ' ',
                        'This function requires the following arguments:',
                        '- num_particles: number of particles to use in the swarm',
                        '- num_generations: number of iterations used by the swarm',
                        '- max_speed: maximum speed of the particles in each direction (in (0, 1])',
                        '- box constraints via key words: constraints are lists [lb, ub]', ' ',
                        'This solver performs num_particles*num_generations function evaluations.'
                        ])(ParticleSwarm)
