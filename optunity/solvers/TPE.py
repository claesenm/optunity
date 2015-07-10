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

from .solver_registry import register_solver
from .util import Solver, _copydoc
import functools

import random

_numpy_available = True
try:
    import numpy
except ImportError:
    _numpy_available = False

_hyperopt_available = True
try:
    import hyperopt
except ImportError:
    _hyperopt_available = False

class TPE(Solver):
    """
    .. include:: /global.rst

    This solver implements the Tree-structured Parzen Estimator, as described in [TPE2011]_.
    This solver uses Hyperopt in the back-end and exposes the TPE estimator with uniform priors.

    Please refer to |tpe| for details about this algorithm.

    .. [TPE2011] Bergstra, James S., et al. "Algorithms for hyper-parameter optimization." Advances in Neural Information Processing Systems. 2011

    """

    def __init__(self, num_evals=100, seed=None, **kwargs):
        """

        Initialize the TPE solver.

        :param num_evals: number of permitted function evaluations
        :type num_evals: int
        :param seed: the random seed to be used
        :type seed: double
        :param kwargs: box constraints for each hyperparameter
        :type kwargs: {'name': [lb, ub], ...}


        """
        if not _hyperopt_available:
            raise ImportError('This solver requires Hyperopt but it is missing.')
        if not _numpy_available:
            raise ImportError('This solver requires NumPy but it is missing.')

        self._seed = seed
        self._bounds = kwargs
        self._num_evals = num_evals

    @staticmethod
    def suggest_from_box(num_evals, **kwargs):
        """
        Verify that we can effectively make a solver from box.

        >>> s = TPE.suggest_from_box(30, x=[0, 1], y=[-1, 0], z=[-1, 1])
        >>> solver = TPE(**s) #doctest:+SKIP

        """
        d = dict(kwargs)
        d['num_evals'] = num_evals
        return d

    @property
    def seed(self):
        return self._seed

    @property
    def bounds(self):
        return self._bounds

    @property
    def num_evals(self):
        return self._num_evals

    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=True, pmap=map):

        if maximize:
            def obj(args):
                kwargs = dict([(k, v) for k, v in zip(self.bounds.keys(), args)])
                return -f(**kwargs)

        else:
            def obj(args):
                kwargs = dict([(k, v) for k, v in zip(self.bounds.keys(), args)])
                return f(**kwargs)

        seed = self.seed if self.seed else random.randint(0, 9999999999)
        algo = functools.partial(hyperopt.tpe.suggest, seed=seed)

        space = [hyperopt.hp.uniform(k, v[0], v[1]) for k, v in self.bounds.items()]
        best = hyperopt.fmin(obj, space=space, algo=algo, max_evals=self.num_evals)
        return best, None


# TPE is a simple wrapper around Hyperopt's TPE solver
if _hyperopt_available and _numpy_available:
    TPE = register_solver('TPE', 'Tree of Parzen estimators',
                        ['TPE: Tree of Parzen Estimators']
                          )(TPE)
