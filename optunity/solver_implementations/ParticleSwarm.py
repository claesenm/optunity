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

import collections
import math
import operator as op
import random
import array
import functools

from ..solver_registry import register_solver
from ..solvers import Solver, _copydoc, uniform_in_bounds

@register_solver('particle swarm',
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
                  ])
class ParticleSwarm(Solver):
    """
    TODO
    """

    class Particle:
        def __init__(self, position, speed, best, fitness, best_fitness):
            """Constructs a Particle."""
            self.position = position
            self.speed = speed
            self.best = best
            self.fitness = fitness
            self.best_fitness = best_fitness

        def clone(self):
            """Clones this Particle."""
            return ParticleSwarm.Particle(position=self.position[:], speed=self.speed[:],
                                          best=self.best[:], fitness=self.fitness,
                                          best_fitness=self.best_fitness)

        def __str__(self):
            string = 'Particle{position=' + str(self.position)
            string += ', speed=' + str(self.speed)
            string += ', best=' + str(self.best)
            string += ', fitness=' + str(self.fitness)
            string += ', best_fitness=' + str(self.best_fitness)
            string += '}'
            return string

    def __init__(self, num_particles, num_generations, max_speed=None, **kwargs):
        """blah"""

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
        self._smin = map(op.neg, self.smax)

        self._phi1 = 2.0
        self._phi2 = 2.0

    @property
    def phi1(self):
        return self._phi1

    @property
    def phi2(self):
        return self._phi2

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
        part = ParticleSwarm.Particle(position=array.array('d', uniform_in_bounds(self.bounds)),
                                      speed=array.array('d', map(random.uniform,
                                                                 self.smin, self.smax)),
                                      best=None, fitness=None, best_fitness=None)
        return part

    def updateParticle(self, part, best, phi1, phi2):
        u1 = (random.uniform(0, phi1) for _ in range(len(part.position)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part.position)))
        v_u1 = map(op.mul, u1,
                    map(op.sub, part.best, part.position))
        v_u2 = map(op.mul, u2,
                    map(op.sub, best.position, part.position))
        part.speed = array.array('d', map(op.add, part.speed,
                                          map(op.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if speed < self.smin[i]:
                part.speed[i] = self.smin[i]
            elif speed > self.smax[i]:
                part.speed[i] = self.smax[i]
        part.position[:] = array.array('d', map(op.add, part.position, part.speed))

    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=True, pmap=map):

        @functools.wraps(f)
        def evaluate(particle):
            return f(**dict([(k, v)
                              for k, v in zip(self.bounds.keys(),
                                              particle.position)]))

        if maximize:
            fit = 1.0
        else:
            fit = -1.0

        pop = [self.generate() for _ in range(self.num_particles)]
        best = None

        for g in range(self.num_generations):
            fitnesses = pmap(evaluate, pop)
            for part, fitness in zip(pop, fitnesses):
                part.fitness = fit*fitness
                if not part.best or part.best_fitness < part.fitness:
                    part.best = part.position
                    part.best_fitness = part.fitness
                if not best or best.fitness < part.fitness:
                    best = part.clone()
            for part in pop:
                self.updateParticle(part, best, self.phi1, self.phi2)

        return dict([(k, v)
                        for k, v in zip(self.bounds.keys(), best.position)]), None
