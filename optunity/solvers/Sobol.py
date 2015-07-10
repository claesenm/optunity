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
import array
import functools
try:
    # Python 2
    from itertools import izip_longest
except ImportError:
    # Python 3
    from itertools import zip_longest as izip_longest

from ..functions import static_key_order
from .solver_registry import register_solver
from .util import Solver, _copydoc, uniform_in_bounds
from . import util

try:
    # Python 2
    irange = irange
except NameError:
    #Python 3
    irange = range

# Parts of this implementation were obtained from here:
# obtained from http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html
# we have removed all dependencies on numpy and replaced with standard
# library functions
# All functions we reused are annotated.

# The MIT License (MIT)
#
# Copyright (c) 2014 John Burkardt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.



@register_solver('sobol',
                 'sample the search space using a Sobol sequence',
                 ['Generates a Sobol sequence of points to sample in the search space.',
                  '',
                  'A Sobol sequence is a low discrepancy quasirandom sequence,',
                  'specifically designed to cover the search space evenly.',
                  '',
                  'Details are available at: http://en.wikipedia.org/wiki/Sobol_sequence',
                  'This sampling method should be preferred over sampling uniformly at random.'])
class Sobol(Solver):
    """
    .. include:: /global.rst

    Please refer to |sobol| for details on this algorithm.
    """


    def __init__(self, num_evals, seed=None, skip=None, **kwargs):
        """
        Initializes a Sobol sequence solver.

        :param num_evals: number of evaluations to use
        :type num_evals: int
        :param skip: the number of initial elements of the sequence to skip, if None a random skip is generated
        :type skip: int or None
        :param kwargs: box constraints for each hyperparameter
        :type kwargs: {'name': [lb, ub], ...}

        The search space is rescaled to the unit hypercube before the solving process begins.

        """

        assert all([len(v) == 2 and v[0] <= v[1]
                    for v in kwargs.values()]), 'kwargs.values() are not [lb, ub] pairs'
        self._bounds = kwargs
        self._num_evals = num_evals
        self._skip = skip if skip else random.randint(200, 1000)


    @_copydoc(Solver.optimize)
    def optimize(self, f, maximize=True, pmap=map):

        sequence = Sobol.i4_sobol_generate(len(self.bounds), self.num_evals, self.skip)
        scaled = list(map(lambda x: util.scale_unit_to_bounds(x, self.bounds.values()),
                          sequence))

        best_pars = None
        @functools.wraps(f)
        def fwrap(args):
            kwargs = dict([(k, v) for k, v in zip(self.bounds.keys(), args)])
            return f(**kwargs)

        if maximize:
            comp = lambda score, best: score > best
        else:
            comp = lambda score, best: score < best

        scores = pmap(fwrap, scaled)
        scores = map(util.score, scores)

        if maximize:
            comp = max
        else:
            comp = min
        best_idx, _ = comp(enumerate(scores), key=op.itemgetter(1))
        best_pars = scaled[best_idx] #op.itemgetter(best_idx)(scaled)
        return dict([(k, v) for k, v in zip(self.bounds.keys(), best_pars)]), None

    @property
    def bounds(self): return self._bounds

    @property
    def num_evals(self): return self._num_evals

    @property
    def skip(self): return self._skip

    @staticmethod
    def suggest_from_box(num_evals, **kwargs):
        """Create a configuration for a Sobol solver.

        :param num_evals: number of permitted function evaluations
        :type num_evals: int
        :param kwargs: box constraints
        :type kwargs: {'param': [lb, ub], ...}


        Verify that we can effectively make a solver from box.

        >>> s = Sobol.suggest_from_box(30, x=[0, 1], y=[-1, 0], z=[-1, 1])
        >>> solver = Sobol(**s)

        """
        d = util.shrink_bounds(kwargs)
        d['num_evals'] = num_evals
        return d

    @staticmethod
    def bitwise_xor(a, b):
        """
        Returns the bitwise_xor of a and b as a bitstring.

        :param a: first number
        :type a: int
        :param b: second number
        :type b: int

        >>> Sobol.bitwise_xor(13, 17)
        28
        >>> Sobol.bitwise_xor(31, 5)
        26

        """
        to_binary = lambda x: bin(x)[2:]
        abin = to_binary(a)[::-1]
        bbin = to_binary(b)[::-1]
        xor = lambda x, y: '0' if (x == y) else '1'
        lst = [xor(x, y) for x, y in izip_longest(abin, bbin, fillvalue='0')]
        lst.reverse()
        return int("".join(lst), 2)

    @staticmethod
    def i4_bit_hi1 ( n ):
        """
        Returns the position of the high 1 bit base 2 in an integer.

        :param n: the integer to be measured
        :type n: int
        :returns: (int) the number of bits base 2

        This was taken from http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html
        Licensing:
            This code is distributed under the MIT license.

        Modified:
            22 February 2011

        Author:
            Original MATLAB version by John Burkardt.
            PYTHON version by Corrado Chisari

        """

        i = math.floor ( n )
        bit = 0
        while ( 1 ):
            if ( i <= 0 ):
                break
            bit += 1
            i = math.floor ( i / 2. )
        return bit

    @staticmethod
    def i4_bit_lo0 ( n ):
        """
        Returns the position of the low 0 bit base 2 in an integer.

        :param n: the integer to be measured
        :type n: int
        :returns: (int) the number of bits base 2

        This was taken from http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html
        Licensing:
            This code is distributed under the MIT license.

        Modified:
            22 February 2011

        Author:
            Original MATLAB version by John Burkardt.
            PYTHON version by Corrado Chisari

        """
        bit = 0
        i = math.floor ( n )
        while ( 1 ):
            bit = bit + 1
            i2 = math.floor ( i / 2. )
            if ( i == 2 * i2 ):
                break

            i = i2
        return bit

    @staticmethod
    def i4_sobol_generate ( m, n, skip ):
        """Generates a Sobol sequence.

        :param m: the number of dimensions (our implementation supports up to 40)
        :type m: int
        :param n: the length of the sequence to generate
        :type n: int
        :param skip: the number of initial elements in the sequence to skip
        :type skip: int
        :returns: a list of length n containing m-dimensional points of the Sobol sequence

        """

        r = [Sobol.i4_sobol(m, seed)[0] for seed in irange(skip, skip + n)]
        return r

    @staticmethod
    def i4_sobol ( dim_num, seed ):
        """
        Generates a new quasi-random Sobol vector with each call.

        :param dim_num: number of dimensions of the Sobol vector
        :type dim_num: int
        :param seed: the seed to use to generate the Sobol vector
        :type seed: int
        :returns: the next quasirandom vector and the next seed to use

        This was taken from http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html
        Licensing:
            This code is distributed under the MIT license.

        Modified:
            22 February 2011

        Author:
            Original MATLAB version by John Burkardt.
            PYTHON version by Corrado Chisari

        """
        global atmost
        global dim_max
        global dim_num_save
        global initialized
        global lastq
        global log_max
        global maxcol
        global poly
        global recipd
        global seed_save
        global v

        if ( not 'initialized' in globals().keys() ):
            initialized = 0
            dim_num_save = -1

        if ( not initialized or dim_num != dim_num_save ):
            initialized = 1
            dim_max = 40
            dim_num_save = -1
            log_max = 30
            seed_save = -1
    #
    #    Initialize (part of) V.
    #
            v = [[0] * dim_max for _ in irange(log_max)]
            v[0][0:40] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]

            v[1][2:40] = [1, 3, 1, 3, 1, 3, 3, 1, \
                3, 1, 3, 1, 3, 1, 1, 3, 1, 3, \
                1, 3, 1, 3, 3, 1, 3, 1, 3, 1, \
                3, 1, 1, 3, 1, 3, 1, 3, 1, 3 ]

            v[2][3:40] = [7, 5, 1, 3, 3, 7, 5, \
                5, 7, 7, 1, 3, 3, 7, 5, 1, 1, \
                5, 3, 3, 1, 7, 5, 1, 3, 3, 7, \
                5, 1, 1, 5, 7, 7, 5, 1, 3, 3 ]

            v[3][5:40] = [1, 7, 9, 13, 11, \
                1, 3, 7, 9, 5, 13, 13, 11, 3, 15, \
                5, 3, 15, 7, 9, 13, 9, 1, 11, 7, \
                5, 15, 1, 15, 11, 5, 3, 1, 7, 9 ]

            v[4][7:40] = [9, 3,27, \
                15,29,21,23,19,11,25, 7,13,17, \
                1,25,29, 3,31,11, 5,23,27,19, \
                21, 5, 1,17,13, 7,15, 9,31, 9 ]

            v[5][13:40] = [37, 33, 7, 5,11, 39, 63, \
                27, 17, 15, 23, 29, 3, 21, 13, 31, 25, \
                9, 49, 33, 19, 29, 11, 19, 27, 15, 25 ]

            v[6][19:40] = [13, \
                33, 115, 41, 79, 17, 29, 119, 75, 73, 105, \
                7, 59, 65, 21, 3, 113, 61, 89, 45, 107 ]

            v[7][37:40] = [7, 23, 39 ]
    #
    #    Set POLY.
    #
            poly= [ \
                1,     3,     7,    11,    13,    19,    25,    37,    59,    47, \
                61,    55,    41,    67,    97,    91, 109, 103, 115, 131, \
                193, 137, 145, 143, 241, 157, 185, 167, 229, 171, \
                213, 191, 253, 203, 211, 239, 247, 285, 369, 299 ]

            atmost = 2**log_max - 1
    #
    #    Find the number of bits in ATMOST.
    #
            maxcol = Sobol.i4_bit_hi1 ( atmost )
    #
    #    Initialize row 1 of V.
    #
    #        v[0,0:maxcol] = 1
            for i in irange(maxcol):
                v[i][0] = 1

    #
    #    Things to do only if the dimension changed.
    #
        if ( dim_num != dim_num_save ):
    #
    #    Check parameters.
    #
            if ( dim_num < 1 or dim_max < dim_num ):
                raise ValueError('I4_SOBOL - Fatal error! The spatial dimension DIM_NUM should satisfy: 1 <= DIM_NUM <= %d, But this input value is DIM_NUM = %d' % (dim_max, dim_num))

            dim_num_save = dim_num
    #
    #    Initialize the remaining rows of V.
    #
            for i in irange(2 , dim_num+1):
    #
    #    The bits of the integer POLY(I) gives the form of polynomial I.
    #
    #    Find the degree of polynomial I from binary encoding.
    #
                j = poly[i-1]
                m = 0
                while ( 1 ):
                    j = math.floor ( j / 2. )
                    if ( j <= 0 ):
                        break
                    m = m + 1
    #
    #    Expand this bit pattern to separate components of the logical array INCLUD.
    #
                j = poly[i-1]
                includ = [0 for _ in irange(m)]
                for k in irange(m, 0, -1):
                    j2 = math.floor ( j / 2. )
                    includ[k-1] =  (j != 2 * j2 )
                    j = j2
    #
    #    Calculate the remaining elements of row I as explained
    #    in Bratley and Fox, section 2.
    #
                for j in irange( m+1, maxcol+1 ):
                    newv = v[j-m-1][i-1]
                    l = 1
                    for k in irange(1, m+1):
                        l = 2 * l
                        if ( includ[k-1] ):
                            newv = Sobol.bitwise_xor ( int(newv), int(l * v[j-k-1][i-1]) )
                    v[j-1][i-1] = newv
    #
    #    Multiply columns of V by appropriate power of 2.
    #
            l = 1
            for j in irange( maxcol-1, 0, -1):
                l = 2 * l
                v[j-1][0:dim_num] = map(lambda x: x * l, v[j-1][0:dim_num])
    #
    #    RECIPD is 1/(common denominator of the elements in V).
    #
            recipd = 1.0 / ( 2 * l )
            lastq = [0 for _ in irange(dim_num)]

        seed = int(math.floor ( seed ))

        if ( seed < 0 ):
            seed = 0

        if ( seed == 0 ):
            l = 1
            lastq = [0 for _ in irange(dim_num)]

        elif ( seed == seed_save + 1 ):
    #
    #    Find the position of the right-hand zero in SEED.
    #
            l = Sobol.i4_bit_lo0 ( seed )

        elif ( seed <= seed_save ):

            seed_save = 0
            l = 1
            lastq = [0 for _ in irange(dim_num)]

            for seed_temp in irange( int(seed_save), int(seed)):
                l = Sobol.i4_bit_lo0 ( seed_temp )
                for i in irange(1 , dim_num+1):
                    lastq[i-1] = Sobol.bitwise_xor ( int(lastq[i-1]), int(v[l-1][i-1]) )

            l = Sobol.i4_bit_lo0 ( seed )

        elif ( seed_save + 1 < seed ):

            for seed_temp in irange( int(seed_save + 1), int(seed) ):
                l = Sobol.i4_bit_lo0 ( seed_temp )
                for i in irange(1, dim_num+1):
                    lastq[i-1] = Sobol.bitwise_xor ( int(lastq[i-1]), int(v[l-1][i-1]) )

            l = Sobol.i4_bit_lo0 ( seed )
    #
    #    Check that the user is not calling too many times!
    #
        if ( maxcol < l ):
            raise ValueError('I4_SOBOL - Fatal error! Too many calls: MAXCOL = %d, L = %d' % (maxcol, l))
    #
    #    Calculate the new components of QUASI.
    #
        quasi = [0 for _ in irange(dim_num)]
        for i in irange( 1, dim_num+1):
            quasi[i-1] = lastq[i-1] * recipd
            lastq[i-1] = Sobol.bitwise_xor ( int(lastq[i-1]), int(v[l-1][i-1]) )

        seed_save = seed
        seed = seed + 1

        return [ quasi, seed ]

    @staticmethod
    def maxdim():
        """The maximum dimensionality that we currently support."""
        return 40

