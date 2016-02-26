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
    import numpy as np
except ImportError:
    _numpy_available = False

_bayesopt_available = True
try:
    import bayesopt
except ImportError:
    _bayesopt_available = False

class BayesOpt(Solver):
    """
    .. include:: /global.rst

    This solver provides an interface to BayesOpt, as described in [BO2014]_.
    This solver uses BayesOpt in the back-end and exposes its solver with uniform priors.

    Please refer to |bayesopt| for details about this algorithm.

    .. note::

        This solver will always output some text upon running.
        This is caused internally by BayesOpt, which provides no way to disable all output.

    .. [BO2014] Martinez-Cantin, Ruben. "BayesOpt: A Bayesian optimization library for nonlinear optimization, experimental design and bandits." The Journal of Machine Learning Research 15.1 (2014): 3735-3739.

    """

    def __init__(self, num_evals=100, seed=None, **kwargs):
        """

        Initialize the BayesOpt solver.

        :param num_evals: number of permitted function evaluations
        :type num_evals: int
        :param seed: the random seed to be used
        :type seed: double
        :param kwargs: box constraints for each hyperparameter
        :type kwargs: {'name': [lb, ub], ...}


        """
        if not _bayesopt_available:
            raise ImportError('This solver requires bayesopt but it is missing.')
        if not _numpy_available:
            raise ImportError('This solver requires NumPy but it is missing.')

        self._seed = seed
        self._bounds = kwargs
        self._num_evals = num_evals
        # bayesopt does not support open intervals,
        # while optunity uses open intervals
        # so we manually shrink the bounding box slightly (yes, this is a dirty fix)
        delta = 0.001
        self._lb = np.array(map(lambda x: float(x[1][0] + delta * (x[1][1] - x[1][0])),
                                sorted(kwargs.items())))
        self._ub = np.array(map(lambda x: float(x[1][1] - delta * (x[1][1] - x[1][0])),
                                sorted(kwargs.items())))

    @staticmethod
    def suggest_from_box(num_evals, **kwargs):
        """
        Verify that we can effectively make a solver from box.

        >>> s = BayesOpt.suggest_from_box(30, x=[0, 1], y=[-1, 0], z=[-1, 1])
        >>> solver = BayesOpt(**s) #doctest:+SKIP

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
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    @property
    def num_evals(self):
        return self._num_evals

    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=True, pmap=map):

        seed = self.seed if self.seed else random.randint(0, 9999)
        params = {'n_iterations': self.num_evals,
                  'random_seed': seed,
                  'n_iter_relearn': 3,
                  'verbose_level': 0}
        n_dimensions = len(self.lb)

        print('lb %s' % str(self.lb))
        print('ub %s' % str(self.ub))

        if maximize:
            def obj(args):
                kwargs = dict([(k, v) for k, v in zip(sorted(self.bounds.keys()), args)])
                return -f(**kwargs)

        else:
            def obj(args):
                kwargs = dict([(k, v) for k, v in zip(sorted(self.bounds.keys()), args)])
                return f(**kwargs)

        mvalue, x_out, error = bayesopt.optimize(obj, n_dimensions,
                                                 self.lb, self.ub, params)
        best = dict([(k, v) for k, v in zip(sorted(self.bounds.keys()), x_out)])
        return best, None


# BayesOpt is a simple wrapper around bayesopt's BayesOpt solver
if _bayesopt_available and _numpy_available:
    BayesOpt = register_solver('BayesOpt', 'Tree of Parzen estimators',
                               ['BayesOpt: Tree of Parzen Estimators']
                               )(BayesOpt)
