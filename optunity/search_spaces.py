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

"""

Functionality to deal with exotic search spaces, with conditional hyperparameters.

A search space is defined as a dictionary mapping strings to nodes. Each node is one of the following:

    1. A new sub search space, that is a new dictionary with the same structure.
    2. 2-element list or tuple, containing (lb, ub) for the associated hyperparameter.
    3. None, to indicate a terminal node that has no numeric value associated to it.

A simple example search space is that for SVM, where we want to optimize the kernel function and (optionally)
its hyperparameterization.

.. code:
    search = {'kernel': {'linear': None,
                        'rbf': {'gamma': [0, 1]},
                        'poly': {'degree': [2, 4]}
                        },
              'c': [0, 1]
             }

Alternatively, it is possible to optimize the regularization parameter 'c' for every choice of kernel separately,
as its required range may be different.

.. code:
    search = {'kernel': {'linear': {'c': [0, 1]},
                         'rbf': {'gamma': [0, 1], 'c': [0, 2]},
                         'poly': {'degree': [2, 4], 'c': [0, 3]}
                        }
             }

Main features in this module:

* :func:`logged`
* :func:`max_evals`

.. moduleauthor:: Marc Claesen
"""

import functools
import itertools
import collections
import math

class Options(object):

    def __init__(self, cases):
        self._cases = cases

    @property
    def cases(self): return self._cases

    def __iter__(self):
        for case in self.cases:
            yield case

    def __repr__(self): return "{%s}" % ", ".join(self.cases)

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx): return self.cases[idx]


class Node(object):

    def __init__(self, key, value):
        self._key = key
        if type(value) is dict:
            self._content = [Node(k, v) for k, v in sorted(value.items())]
        else: self._content = value

    @property
    def key(self): return self._key

    @property
    def content(self): return self._content

    @property
    def terminal(self):
        if self.content:
            return not type(self.content[0]) == type(self)
        return True

    @property
    def choice(self):
        """Determines whether this node is a choice."""
        return self.content is None

    def __iter__(self):
        if self.terminal:
            yield self.key, self.content
        else:
            content = list(itertools.chain(*self.content))

            if any([not x.terminal or x.choice for x in self.content]):
                content.insert(0, ([], Options([node.key for node in self.content])))

            for k, v in content:
                if type(k) is list: key = [self.key] + k
                else: key = [self.key, k]
                yield key, v


class SearchTree(object):

    def __init__(self, d):
        self._content = [Node(k, v) for k, v in sorted(d.items())]
        self._vectordict = collections.OrderedDict()
        self._vectorcontent = collections.OrderedDict()

    @property
    def vectordict(self): return self._vectordict

    @property
    def vectorcontent(self): return self._vectorcontent

    @property
    def content(self): return self._content

    def __iter__(self):
        for i in self.content:
            for k, v in i:
                yield k, v

    def to_box(self):
        if not self.vectordict:
            for k, v in self:
                key = '-'.join(k)
                if type(v) is Options:
                    self.vectordict[key] = [0.0, float(len(v))]
                    self.vectorcontent[key] = v
                elif v is None:
                    pass
                else:
                    self.vectordict[key] = v
                    self.vectorcontent[key] = v

        return dict([(k, v) for k, v in self.vectordict.items()])

    def decode(self, vd):
        result = {}
        currently_decoding_nested = []
        items = sorted(vd.items())
        idx = 0

        while idx < len(items):

            k, v = items[idx]
            keylist = k.split('-')

            if currently_decoding_nested and len(keylist) >= len(currently_decoding_nested):
                if not all(map(lambda t: t[0] == t[1], zip(currently_decoding_nested, keylist))):
                    # keylist doesnt match what we are currently decoding
                    # and it is longer than what we are decoding
                    # so this must be the wrong key, skip it
                    idx += 1

                    # add a None value for this particular function argument
                    # this is required to keep call logs functional
                    key = keylist[-1]
                    if not key in result: result[key] = None
                    continue

            elif currently_decoding_nested:
                # keylist is shorter than decoding list -> move up one nesting level
                currently_decoding_nested = currently_decoding_nested[:-2]
                continue

            content = self.vectorcontent[k]
            if type(content) is Options:
                # determine which option to use
                # this is done by checking in which partition the current value
                # of the choice parameter (v) falls

                option_idx = int(math.floor(v))
                option = content[option_idx]
                result["-".join(keylist[len(currently_decoding_nested)-1:])] = option
                currently_decoding_nested.extend([k, option])

                idx += 1

            else:
                result[keylist[-1]] = v
                idx += 1

        return result

    def wrap_decoder(self, f):
        """Wraps a function to automatically decode arguments based on given SearchTree."""
        @functools.wraps(f)
        def wrapped(**kwargs):
            decoded = self.decode(kwargs)
            return f(**decoded)
        return wrapped





#hpars = {'kernel': {'linear': {'c': [0, 1]},
#                    'rbf': {'gamma': [0, 1], 'c': [0, 10]},
#                    'poly': {'degree': [2, 4], 'c': [0, 2]}
#                   }
#        }

#hpars = {'kernel': {'linear': None,
#                    'rbf': {'gamma': [0, 1]},
#                    'poly': {'degree': [2, 4]}
#                    },
#            'c': [0, 1]
#            }

#hpars = {'kernel': {'linear': {'c': [0, 1]},
#                    'rbf': {'gamma': [0, 1], 'c': [0, 10]},
#                    'poly': {'degree': [2, 4], 'c': [0, 2]},
#                    'choice': {'choice1': None, 'choice2': None}
#                   }
#        }

#tree = SearchTree(hpars)
#l = list(tree)
#v = tree.to_box()
#v2 = v.copy()
#v2['kernel'] = 3.5
#v2['kernel-choice'] = 0.2
#tree.decode(v2)

