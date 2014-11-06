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
import operator as op
import random
import itertools

from ..solvers import Solver, _copydoc, uniform_in_bounds


#@register_solver('annealing',
#                 'coupled simulated annealing',
#                 ['TODO'])
class CSA(Solver):
    """
    TODO

    Based on ftp://ftp.esat.kuleuven.be/pub/SISTA/sdesouza/papers/CSA2009accepted.pdf
    """

    class SA:
        """Class to model a single simulated annealing process."""

        def __init__(self, x0):
            self._x = x0[:]
            self._y = x0[:]
            self._cost = None
            self._ycost = None
            self._log = []

        @property
        def log(self):
            """Returns a log of all solutions this process has been in.

            This is returned as a list of (solution, cost)-tuples."""
            return self._log

        @property
        def T(self):
            return self._T

        @property
        def Tacc(self):
            return self._Tacc

        @property
        def k(self):
            return self._k

        @property
        def maxk(self):
            return self._maxk

        @property
        def y(self):
            return self._y

        @property
        def ycost(self):
            return self._ycost

        @ycost.setter
        def ycost(self, value):
            self._ycost = value

        @property
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value[:]

        @property
        def cost(self):
            return self._cost

        def generate(self, gk, Tk):
            """Generate a random probe solution."""
            eps = gk(Tk)
            self._y = map(op.add, self.x, eps)

        def accept(self, accept_probability):
            # update the log of this SA process
            self._log.append((self.y[:], self.ycost))

            # accept if required
            if random.uniform(0, 1) <= accept_probability:
                self.x = self.y
                self._cost = self.ycost
                self._y = None
                self._ycost = None
            else:
                self._y = None
                self._ycost = None

    def __init__(self, num_processes, num_generations, T_0, Tacc_0, **kwargs):
        """blah"""
        assert all([len(v) == 2 and v[0] <= v[1]
                    for v in kwargs.values()]), 'kwargs.values() are not [lb, ub] pairs'
        self._bounds = kwargs
        self._num_dimensions = len(kwargs)
        self._num_processes = num_processes
        self._num_generations = num_generations
        self._T_0 = T_0
        self._Tacc_0 = Tacc_0

    def g(self, Tk):
        return [Tk * CSA.cauchy_sample() for _ in range(self.num_dimensions)]

    @property
    def num_dimensions(self):
        return self._num_dimensions

    @property
    def num_processes(self):
        return self._num_processes

    @property
    def num_generations(self):
        return self._num_generations

    def T(self, k=0):
        return self._T_0 / (k + 1)

    def Tacc(self, k=0):
        if k:
            return self._Tacc_0 / math.log(k + 1)
        else:
            return self._Tacc_0

    def exp(self, cost, multiplier, k):
        return math.exp(-cost * multiplier / self.Tacc(k))

    @staticmethod
    def suggest_from_box(num_evals, **kwargs):
        d = dict(kwargs)
        if num_evals > 200:
            d['num_processes'] = 50
            d['num_generations'] = int(math.ceil(float(num_evals) / 50))
        elif num_evals > 10:
            d['num_processes'] = 10
            d['num_generations'] = int(math.ceil(float(num_evals) / 10))
        else:
            d['num_processes'] = num_evals
            d['num_generations'] = 1
        return d

    @staticmethod
    def cauchy_sample():
        return math.tan(math.pi * (random.uniform(0, 1) - 0.5))

    @property
    def bounds(self):
        return self._bounds

    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=True, pmap=map):
        if maximize:
            mult = -1.0
            comp = op.gt
        else:
            mult = 1.0
            comp = op.lt

        def evaluate(state):
            print('state: ' + str(state))
            return f(**dict([(k, v)
                              for k, v in zip(self.bounds.keys(),
                                              state)]))

        # initialization
        processes = [CSA.SA(uniform_in_bounds(self.bounds))
                     for _ in range(self.num_processes)]

        states = [p.y for p in processes]
        costs = pmap(evaluate, states)

        for process, cost in zip(processes, costs):
            process.ycost = cost
            process.accept(1.0)

        # start of annealing iterations
        for iteration in range(self.num_generations - 1):

            k = float(iteration)
            for process in processes:
                process.generate(self.g, self.T(k))

            states = [p.y for p in processes]
            costs = pmap(evaluate, states)

            gamma = reduce(op.add, [self.exp(proc.cost, mult, k)
                                    for proc in processes])

            print('gamma: ' + str(gamma))
            for process, ycost in zip(processes, costs):
                process.ycost = ycost
                if comp(process.ycost, process.cost):
                    accept_probability = 1.0
                else:
                    expcost = self.exp(ycost, mult, k)
                    accept_probability = expcost / (expcost + gamma)
                print('cost: ' + str(process.cost) + ' ycost: ' + str(process.ycost) + ' prob: ' + str(accept_probability))
                process.accept(accept_probability)

        logs = itertools.chain(*[p.log for p in processes])
        if maximize:
            best = max(logs, key=op.itemgetter(1))[0]
        else:
            best = min(logs, key=op.itemgetter(1))[0]

        return dict([(k, v)
                        for k, v in zip(self.bounds.keys(), best)]), None
