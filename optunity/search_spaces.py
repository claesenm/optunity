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

TODO

* :func:`logged`
* :func:`max_evals`

.. moduleauthor:: Marc Claesen
"""

import functools
import itertools
import collections
import math

DELIM = '|'

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
    """Models a node within a search space.

    Nodes can be internal or terminal, and may or may not be choices.
    A choice is a node that models a discrete choice out of k > 1 options.

    """

    def __init__(self, key, value):
        self._key = key
        if type(value) is dict:
            self._value = [Node(k, v) for k, v in sorted(value.items())]
        else: self._value = value

    @property
    def key(self): return self._key

    @property
    def value(self): return self._value

    @property
    def terminal(self):
        """Returns whether or not this Node is terminal.

        A terminal node has a non-dictionary value (numeric, list or None).
        """
        if self.value:
            return not type(self.value[0]) == type(self)
        return True

    @property
    def choice(self):
        """Determines whether this node is a choice.

        A choice is a node that models a discrete choice out of k > 1 options.
        """
        return self.value is None

    def __iter__(self):
        """Iterates over this node.

        If the node is terminal, yields the key and value.
        Otherwise, first yields all values and then iterates over the values.

        """
        if self.terminal:
            yield self.key, self.value
        else:
            value = list(itertools.chain(*self.value))

            if any([not x.terminal or x.choice for x in self.value]):
                value.insert(0, ([], Options([node.key for node in self.value])))

            for k, v in value:
                if type(k) is list: key = [self.key] + k
                else: key = [self.key, k]
                yield key, v


class SearchTree(object):
    """Tree structure to model a search space.

    Fairly elaborate unit test.

    >>> space = {'a': {'b0': {'c0': {'d0': {'e0': [0, 10], 'e1': [-2, -1]},
    ...                              'd1': {'e2': [-3, -1]},
    ...                              'd2': None
    ...                              },
    ...                       'c1': [0.0, 1.0],
    ...                      },
    ...                'b1': {'c2': [-2.0, -1.0]},
    ...                'b2': None
    ...               }
    ...          }
    >>> tree = SearchTree(space)
    >>> b = tree.to_box()
    >>> print(b['a'] == [0.0, 3.0] and
    ...       b['a|b0|c0'] == [0.0, 3.0] and
    ...       b['a|b0|c1'] == [0.0, 1.0] and
    ...       b['a|b1|c2'] == [-2.0, -1.0] and
    ...       b['a|b0|c0|d0|e0'] == [0, 10] and
    ...       b['a|b0|c0|d0|e1'] == [-2, -1] and
    ...       b['a|b0|c0|d1|e2'] == [-3, -1])
    True

    >>> d = tree.decode({'a': 2.5})
    >>> d['a'] == 'b2'
    True

    >>> d = tree.decode({'a': 1.5, 'a|b1|c2': -1.5})
    >>> print(d['a'] == 'b1' and
    ...       d['c2'] == -1.5)
    True

    >>> d = tree.decode({'a': 0.5, 'a|b0|c0': 1.7, 'a|b0|c0|d1|e2': -1.2})
    >>> print(d['a'] == 'b0' and
    ...       d['c0'] == 'd1' and
    ...       d['e2'] == -1.2)
    True

    >>> d = tree.decode({'a': 0.5, 'a|b0|c0': 2.7})
    >>> print(d['a'] == 'b0' and
    ...       d['c0'] == 'd2')
    True

    >>> d = tree.decode({'a': 0.5, 'a|b0|c0': 0.7, 'a|b0|c0|d0|e0': 2.3, 'a|b0|c0|d0|e1': -1.5})
    >>> print(d['a'] == 'b0' and
    ...       d['c0'] == 'd0' and
    ...       d['e0'] == 2.3 and
    ...       d['e1'] == -1.5)
    True

    """

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
                key = DELIM.join(k)
                if type(v) is Options:
                    if len(v) > 1: # options of length one aren't really options
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
            keylist = k.split(DELIM)

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
                result[DELIM.join(keylist[len(currently_decoding_nested):])] = option
                currently_decoding_nested.extend([keylist[-1], option])

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


#hpars = {'algorithm': {'k-nn': {'k': [1, 10]},
#                       'SVM': {'kernel': {'linear': {'C': [0, 2]},
#                                           'rbf': {'gamma': [0, 1], 'C': [0, 10]},
#                                           'poly': {'degree': [2, 5], 'C': [0, 50], 'coef0': [0, 1]}
#                                          }
#                               },
#                       'naive-bayes': None,
#                       'random-forest': {'n_estimators': [100, 300], 'max_features': [5, 100]}
#                       }
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

#tree.decode(v2)

#tree = SearchTree(hpars)
#l = list(tree)
#print('============================')
#print('list')
#print("\n".join(map(str, l)))
#v = tree.to_box()
#print('============================')
#print('box')
#print("\n".join(map(str, v.items())))
#v2 = v.copy()
#v2['kernel'] = 3.5
#v2['kernel-choice'] = 0.2

