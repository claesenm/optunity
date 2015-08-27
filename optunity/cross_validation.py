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

"""This module contains various provisions for cross-validation.

The main functions in this module are:

* :func:`cross_validated`
* :func:`generate_folds`
* :func:`strata_by_labels`
* :func:`random_permutation`
* :func:`mean`
* :func:`identity`
* :func:`list_mean`

.. moduleauthor:: Marc Claesen

"""

import math
import random
import itertools as it
import functools
import operator as op
import array
import inspect


__all__ = ['select', 'random_permutation', 'cross_validated',
           'generate_folds', 'strata_by_labels', 'mean', 'identity',
           'list_mean', 'mean_and_list']

_spark_available = True
try:
    import pyspark
except ImportError:
    _spark_available = False


def select(collection, indices):
    """Selects the subset specified by indices from collection.

    >>> select([0, 1, 2, 3, 4], [1, 3])
    [1, 3]

    """
    try:
        return collection[indices, ...]
    except IndexError: # caused by scipy.sparse in some versions
        return collection[indices, :]
    except TypeError: # not dealing with numpy or comparable, probably a list
        if _spark_available and isinstance(collection, pyspark.rdd.RDD):
            indexset = set(indices)
            return collection.zipWithIndex().filter(lambda x: x[1] in indexset).map(lambda x: x[0])
        return [collection[i] for i in indices]


# https://docs.python.org/2/library/itertools.html#itertools.permutations
def random_permutation(data):
    """Returns a list containing a random permutation of ``r`` elements out of
    ``data``.

    :param data: an iterable containing the elements to permute over
    :returns: returns a list containing permuted entries of ``data``.

    """
    d = data[:]
    random.shuffle(d)
    return d


def strata_by_labels(labels):
    """Constucts a list of strata (lists) based on unique values of ``labels``.

    :param labels: iterable, identical values will end up in identical strata
    :returns: the strata, as a list of lists
    """
    return [list(zip(*g)[0])
            for _, g in it.groupby(enumerate(labels), op.itemgetter(1))]

def _fold_sizes(num_rows, num_folds):
    """Generates fold sizes to partition specified number of rows into number of folds.

    :param num_rows: number of rows (instances)
    :type num_rows: integer
    :param num_folds: number of folds
    :type num_folds: integer
    :returns: an array.array of fold sizes (of length num_folds)

    """
    sizes = array.array('i', [0] * num_folds)
    for i in range(num_folds):
        sizes[i] = int(math.floor(float(num_rows - sum(sizes)) / (num_folds - i)))
    return sizes


def generate_folds(num_rows, num_folds=10, strata=None, clusters=None):
    """Generates folds for a given number of rows.

    :param num_rows: number of data instances
    :param num_folds: number of folds to use (default 10)
    :param strata: (optional) list of lists to indicate different
        sampling strata. Not all rows must be in a stratum. The number of
        rows per stratum must be larger than or equal to num_folds.
    :param clusters: (optional) list of lists indicating clustered instances.
        Clustered instances must be placed in a single fold to avoid
        information leaks.
    :returns: a list of folds, each fold is a list of instance indices

    >>> folds = generate_folds(num_rows=6, num_folds=2, clusters=[[1, 2]], strata=[[3,4]])
    >>> len(folds)
    2
    >>> i1 = [idx for idx, fold in enumerate(folds) if 1 in fold]
    >>> i2 = [idx for idx, fold in enumerate(folds) if 2 in fold]
    >>> i1 == i2
    True
    >>> i3 = [idx for idx, fold in enumerate(folds) if 3 in fold]
    >>> i4 = [idx for idx, fold in enumerate(folds) if 4 in fold]
    >>> i3 == i4
    False

    .. warning::
        Instances in strata are not necessarily spread out over all folds. Some
        folds may already be full due to clusters. This effect should be negligible.

    """

    # sizes per fold and initialization of folds
    sizes = _fold_sizes(num_rows, num_folds)
    folds = [[] for _ in range(num_folds)]

    # the folds that still need to be filled
    fill_queue = set(range(num_folds))

    # the instances that still need to be assigned
    instances = set(range(num_rows))

    if not strata:
        strata = []

    if not clusters:
        clusters = []

    # instances not in any stratum/cluster are treated as a final stratum
    if strata:
        assigned = set(it.chain(*strata))
    else:
        assigned = set()
    if clusters:
        assigned.update(it.chain(*clusters))

    if assigned:
        strata.append(list(filter(lambda x: x not in assigned, instances)))
    else:
        strata.append(list(instances))

    if clusters:
        # sort clusters by size
        sorted_cluster_indices, _ = zip(*sorted(enumerate(clusters),
                                                key=lambda x: len(x[1]),
                                                reverse=True))

        # assign clusters
        for cluster_idx in sorted_cluster_indices:
            # retrieve eligible folds: folds that will not surpass maximum
            # when we assign given cluster to them
            cluster = clusters[cluster_idx]
            cluster_size = len(cluster)
            eligible = list(filter(lambda x: len(folds[x]) + cluster_size <= sizes[x],
                                   fill_queue))

            if not eligible:
                raise ValueError('Unable to assign all clusters to folds.')

            # choose a fold at random
            fold_idx = random.choice(eligible)
            folds[fold_idx].extend(cluster)

            # update instances to-be-assigned
            instances.difference_update(cluster)

            # remove fold from fill_queue if it is full
            if len(folds[fold_idx]) >= sizes[fold_idx]:
                fill_queue.remove(fold_idx)

    # assign strata
    for stratum in strata:
        permuted_stratum = random_permutation(list(filter(lambda x: x in
                                                          instances, stratum[:])))
        while permuted_stratum:
            eligible = list(filter(lambda x: len(folds[x]) < sizes[x], fill_queue))

            if not eligible:
                raise ValueError('Unable to assign all instances to folds.')

            eligible = random_permutation(eligible)
            for instance_idx, fold_idx in zip(permuted_stratum[:], eligible):
                folds[fold_idx].append(instance_idx)
                if len(folds[fold_idx]) >= sizes[fold_idx]:
                    fill_queue.remove(fold_idx)
                instances.remove(instance_idx)

            permuted_stratum = permuted_stratum[len(eligible):]

    return folds

def mean(x):
    return float(sum(x)) / len(x)

def mean_and_list(x):
    """Returns a tuple, the first element is the mean of x and the second is x itself.

    This function can be used as an aggregator in :func:`cross_validated`,

    >>> mean_and_list([1,2,3])
    (2.0, [1, 2, 3])

    """
    return (mean(x), x)

def list_mean(list_of_measures):
    """Computes means of consequent elements in given list.

    :param list_of_measures: a list of tuples to compute means from
    :type list_of_measures: list
    :returns: a list containing the means

    This function can be used as an aggregator in :func:`cross_validated`,
    when multiple performance measures are being returned by the wrapped function.

    >>> list_mean([(1, 4), (2, 5), (3, 6)])
    [2.0, 5.0]

    """
    return list(map(mean, zip(*list_of_measures)))


def identity(x):
    return x

class cross_validated_callable(object):
    """Function decorator that takes care of cross-validation.
    Evaluations of the decorated function will always return a cross-validated
    estimate of generalization performance.

    :param x: data to be used for cross-validation
    :param num_folds: number of cross-validation folds (default 10)
    :param y: (optional) labels to be used for cross-validation.
        If specified, len(labels) must equal len(x)
    :param strata: (optional) strata to account for when generating folds.
        Strata signify instances that must be spread across folds.
        Not every instance must be in a stratum.
        Specify strata as a list of lists of instance indices.
    :param folds: (optional) prespecified cross-validation folds to be used (list of lists (iterations) of lists (folds)).
    :param num_iter: (optional) number of iterations to use (default 1)
    :param regenerate_folds: (optional) whether or not to regenerate folds on every evaluation (default false)
    :param clusters: (optional) clusters to account for when generating folds.
        Clusters signify instances that must be assigned to the same fold.
        Not every instance must be in a cluster.
        Specify clusters as a list of lists of instance indices.
    :param aggregator: function to aggregate scores of different folds (default: mean)

    Use :func:`cross_validated` to create instances of this class.
    """
    def __init__(self, f, x, num_folds=10, y=None, strata=None, folds=None,
                 num_iter=1, regenerate_folds=False, clusters=None,
                 aggregator=mean):
        self._x = x
        self._y = y
        self._strata = strata
        self._clusters = clusters
        self._regenerate_folds = regenerate_folds
        self._f = f
        self._reduce = aggregator
        if folds:
            assert (len(folds) == num_iter), 'Number of fold sets does not equal num_iter.'
            assert (len(folds[0]) == num_folds), 'Number of folds does not match num_folds.'
            self._folds = folds
        else:
            self._folds = [generate_folds(self.len_x, num_folds, self.strata, self.clusters)
                           for _ in range(num_iter)]
        functools.update_wrapper(self, f)

    @property
    def len_x(self):
        """ Number of samples in x """
        if _spark_available and isinstance(self._x, pyspark.rdd.RDD):
            return self._x.count()
        else:
            try:
                return len(self._x)
            except TypeError:
                return self._x.shape[0]


    @property
    def reduce(self):
        """The aggregation function."""
        return self._reduce

    @property
    def f(self):
        """The decorated function."""
        return self._f

    @property
    def folds(self):
        """The cross-validation folds."""
        return self._folds

    @property
    def strata(self):
        """Strata that were used to compute folds."""
        return self._strata

    @property
    def clusters(self):
        """Clusters that were used to compute folds."""
        return self._clusters

    @property
    def x(self):
        """The data that is used for cross-validation."""
        return self._x

    @property
    def y(self):
        """The labels that are used for cross-validation."""
        return self._y

    @property
    def num_folds(self):
        """Number of cross-validation folds."""
        return len(self.folds[0])

    @property
    def num_iter(self):
        """Number of cross-validation iterations."""
        return len(self.folds)

    @property
    def regenerate_folds(self):
        """Whether or not folds are regenerated for each function evaluation."""
        return self._regenerate_folds

    def __call__(self, *args, **kwargs):
        if args:
            # called with positionals, we must translate them to kwargs of f
            # otherwise positionals mess up with bounded arguments
            argspec = inspect.getargspec(self.f).args
            argspec.remove('x_train')
            argspec.remove('x_test')
            if self.y is not None:
                argspec.remove('y_train')
                argspec.remove('y_test')
            for argname, arg in zip(argspec, args):
                kwargs[argname] = arg

        if self.regenerate_folds:
            self._folds = [generate_folds(self.len_x, self.num_folds, self.strata)
                           for _ in range(self.num_iter)]
        scores = []
        for folds in self.folds:
            for fold in range(self.num_folds):
                rows_test = folds[fold]
                rows_train = list(it.chain(*[folds[i]
                                                    for i in range(self.num_folds)
                                                    if not i == fold]))
                kwargs['x_train'] = select(self.x, rows_train)
                kwargs['x_test'] = select(self.x, rows_test)
                if not self.y is None:  # dealing with a supervised algorithm
                    kwargs['y_train'] = select(self.y, rows_train)
                    kwargs['y_test'] = select(self.y, rows_test)
                scores.append(self.f(**kwargs))
        return self.reduce(scores)

    def __getattr__(self, name):
        # http://stackoverflow.com/a/11195604
        if name.startswith('func_'):
            return getattr(self.f, name)
        raise AttributeError


def cross_validated(x, num_folds=10, y=None, strata=None, folds=None, num_iter=1,
                    regenerate_folds=False, clusters=None, aggregator=mean):
    """Function decorator to perform cross-validation as configured.

    :param x: data to be used for cross-validation
    :param num_folds: number of cross-validation folds (default 10)
    :param y: (optional) labels to be used for cross-validation.
        If specified, len(labels) must equal len(x)
    :param strata: (optional) strata to account for when generating folds.
        Strata signify instances that must be spread across folds.
        Not every instance must be in a stratum.
        Specify strata as a list of lists of instance indices.
    :param folds: (optional) prespecified cross-validation folds to be used (list of lists (iterations) of lists (folds)).
    :param num_iter: (optional) number of iterations to use (default 1)
    :param regenerate_folds: (optional) whether or not to regenerate folds on every evaluation (default false)
    :param clusters: (optional) clusters to account for when generating folds.
        Clusters signify instances that must be assigned to the same fold.
        Not every instance must be in a cluster.
        Specify clusters as a list of lists of instance indices.
    :param aggregator: function to aggregate scores of different folds (default: mean)
    :returns: a :class:`cross_validated_callable` with the proper configuration.

    This resulting decorator must be used on a function with the following signature (+ potential other arguments):

    :param x_train: training data
    :type x_train: iterable
    :param y_train: training labels (optional)
    :type y_train: iterable
    :param x_test: testing data
    :type x_test: iterable
    :param y_test: testing labels (optional)
    :type y_test: iterable

    `y_train` and `y_test` must be available of the `y` argument to this function is not None.

    These 4 keyword arguments will be bound upon decoration.
    Further arguments will remain free (e.g. hyperparameter names).

    >>> data = list(range(5))
    >>> @cross_validated(x=data, num_folds=5, folds=[[[i] for i in range(5)]], aggregator=identity)
    ... def f(x_train, x_test, a):
    ...     return x_test[0] + a
    >>> f(a=1)
    [1, 2, 3, 4, 5]
    >>> f(1)
    [1, 2, 3, 4, 5]
    >>> f(a=2)
    [2, 3, 4, 5, 6]

    The number of folds must be less than or equal to the size of the data.

    >>> data = list(range(5))
    >>> @cross_validated(x=data, num_folds=6) #doctest:+SKIP
    ... def f(x_train, x_test, a):
    ...     return x_test[0] + a
    AssertionError

    The number of labels (if specified) must match the number of data instances.

    >>> data = list(range(5))
    >>> labels = list(range(3))
    >>> @cross_validated(x=data, y=labels, num_folds=2) #doctest:+SKIP
    ... def f(x_train, x_test, a):
    ...     return x_test[0] + a
    AssertionError

    """
    assert(num_folds <= len(x))
    assert(y is None or len(y) == len(x))
    args = dict(locals())
    def wrapper(f):
        args['f'] = f
        return cross_validated_callable(**args)
    return wrapper


if __name__ == '__main__':
    pass
