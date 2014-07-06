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


import math
import random
import numpy as np
import itertools
import functools
import collections


def select(collection, indices):
    """Selects the subset specified by indices from collection."""
    if type(collection) is list:
        return [collection[i] for i in indices]
    else:
        # assume np array, TODO: make proper type checking
        return collection[indices]


def map_clusters(clusters):
    """Maps data instance indices to cluster indices."""
    idx2cluster = collections.defaultdict(collections.deque)
    if clusters:
        for cluster, indices in enumerate(clusters):
            for index in indices:
                idx2cluster[index].append(cluster)
    return idx2cluster


# TODO: implement support for clusters
def generate_folds(num_rows, num_folds=10, strata=None, clusters=None,
                   idx2cluster=None):
    """Generates folds for a given number of rows.

    strata is an optional list of lists to indicate different
    sampling strata. Not all rows must be in a stratum. The number of
    rows per stratum must be larger than or equal to num_folds.

    Returned folds are not necessarily of the same size, but should be
    comparable. Size differences may grow for large numbers of strata.
    """

    # FIXME: current partitioning has some issues
    # fold sizes not necessarily (close to) equal
    # if many small strata exist
    def get_folds(rows, num_folds, require=False):
        """Partitions rows in num_folds.

        If require is true, len(rows) > num_folds or an exception is raised.
        """
        if require and num_folds > len(rows):
            raise ValueError

        folds = [None] * num_folds
        idx = np.random.permutation(rows).tolist()
        for fold in range(num_folds):
            fold_size = int(len(idx) / (num_folds - fold))
            folds[fold] = idx[:fold_size]
            del idx[:fold_size]

        # permute so the largest folds are not always at the back
        perm = np.random.permutation(len(folds))
        return [folds[idx] for idx in perm]

    if clusters and not idx2cluster:
        idx2cluster = map_clusters(clusters)

    if strata:  # stratified cross-validation
        ## keep lists per stratum + 1 extra for no-care instances
        permuted_strata = map(np.random.permutation, strata)
        nocares = list(set(range(num_rows)) - set(itertools.chain(*strata)))

        # we  do not require points of each stratum in each fold,
        # since strata can be smaller than number of folds
        folds_per_stratum = map(lambda x: get_folds(x, num_folds, False),
                                permuted_strata)
        # nocare points need not be in every fold
        folds_nocare = get_folds(nocares, num_folds, False)

        # merge folds across strata
        folds_per_stratum.append(folds_nocare)
        folds = [list(itertools.chain(*x)) for x in zip(*folds_per_stratum)]

    else:  # no stratification required
        folds = get_folds(range(num_rows), num_folds, True)

    return folds


class cross_validated_callable(object):
    def __init__(self, f, x, num_folds=10, y=None, strata=None, folds=None,
                 num_iter=1, regenerate_folds=False, clusters=None):
        self._x = x
        self._y = y
        self._strata = strata
        self._clusters = clusters
        self._idx2cluster = map_clusters(clusters)
        # TODO: sanity check between strata & clusters? define what is allowed
        self._regenerate_folds = regenerate_folds
        self._f = f
        if folds:
            assert (len(folds) == num_iter), 'Number of fold sets does not equal num_iter.'
            assert (len(folds[0] == num_folds)), 'Number of folds does not match num_folds.'
            self._folds = folds
        else:
            self._folds = [generate_folds(len(x), num_folds, self.strata,
                                          self.clusters)
                           for _ in range(num_iter)]
        functools.update_wrapper(self, f)

    @property
    def f(self):
        return self._f

    @property
    def folds(self):
        return self._folds

    @property
    def strata(self):
        return self._strata

    @property
    def clusters(self):
        return self._clusters

    @property
    def idx2cluster(self):
        """A mapping of data instances to cluster ids.

        cluster ids are stored in a list, since single
        instances may occur in several clusters."""
        return self._idx2cluster

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def num_folds(self):
        return len(self.folds[0])

    @property
    def num_iter(self):
        return len(self.folds)

    @property
    def regenerate_folds(self):
        return self._regenerate_folds

    def __call__(self, *args, **kwargs):
        if self.regenerate_folds:
            self._folds = [generate_folds(len(x), self.num_folds, self.strata)
                           for _ in range(self.num_iter)]
        scores = []
        for folds in self.folds:
            for fold in range(self.num_folds):
                rows_test = folds[fold]
                rows_train = list(itertools.chain(*[folds[i]
                                                    for i in range(self.num_folds)
                                                    if not i == fold]))
                kwargs['x_train'] = select(self.x, rows_train)
                kwargs['x_test'] = select(self.x, rows_test)
                if not self.y is None:  # dealing with a supervised algorithm
                    kwargs['y_train'] = select(self.y, rows_train)
                    kwargs['y_test'] = select(self.y, rows_test)
                scores.append(self.f(*args, **kwargs))
        return float(sum(scores)) / (self.num_iter * len(scores))


def cross_validated(x, num_folds=10, y=None, strata=None, folds=None, num_iter=1,
                    regenerate_folds=False, clusters=None):
    def wrapper(f):
        cv_callable = cross_validated_callable(f, x, num_folds, y, strata, folds,
                                               num_iter, regenerate_folds, clusters)
        return cv_callable
    return wrapper


if __name__ == '__main__':
    x = list(range(10))
    clusters = [[1,2],[3,4,5]]

    @cross_validated(x, num_folds=3, num_iter=2, clusters=clusters)
    def f1(woops, x_train, x_test):
        '''floopsie docstring'''
        print(x_train)
        return 0

    print(f1.folds)
    result = f1(woops='blah')
    print(result)

    def f(x, y, z):
        return x + y + z

    def f2(**kwargs):
        kwargs['z'] = 2
        return f(**kwargs)
    print(str(f2(x=1, y=2)))

    @cross_validated(list(range(20)), num_folds=10, num_iter=2, strata=[[1,2,3],[6,7,8,9]])
    def f1(woops, x_train, x_test):
        '''floopsie docstring'''
        print(x_train)
        return 0
