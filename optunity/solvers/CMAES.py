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

import math
import functools

from .solver_registry import register_solver
from .util import Solver, _copydoc
from . import util

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


class CMA_ES(Solver):
    """
    .. include:: /global.rst

    Please refer to |cmaes| for details about this algorithm.

    This solver uses an implementation available in the DEAP library [DEAP2012]_.

    .. warning:: This solver has dependencies on DEAP_ and NumPy_
        and will be unavailable if these are not met.

        .. _DEAP: https://code.google.com/p/deap/
        .. _NumPy: http://www.numpy.org

    """

    def __init__(self, num_generations, sigma=1.0, Lambda=None, **kwargs):
        """blah

        .. warning:: |warning-unconstrained|

        """
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
        """Verify that we can effectively make a solver.
        The doctest has to be skipped from automated builds, because DEAP may not be available
        and yet we want documentation to be generated.

        >>> s = CMA_ES.suggest_from_seed(30, x=1.0, y=-1.0, z=2.0)
        >>> solver = CMA_ES(**s) #doctest:+SKIP

        """
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

    @_copydoc(Solver.optimize)
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
            strategy = deap.cma.Strategy(centroid=list(self.start.values()),
                                            sigma=self.sigma, lambda_=self.lambda_)
        else:
            strategy = deap.cma.Strategy(centroid=list(self.start.values()),
                                            sigma=self.sigma)
        toolbox.register("generate", strategy.generate, Individual)
        toolbox.register("update", strategy.update)

        @functools.wraps(f)
        def evaluate(individual):
            return (util.score(f(**dict([(k, v)
                                for k, v in zip(self.start.keys(),
                                                individual)]))),)
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
    CMA_ES = register_solver('cma-es', 'covariance matrix adaptation evolutionary strategy',
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
