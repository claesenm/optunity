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
import operator as op

def contingency_tables(ys, decision_values, positive=True, presorted=False):
    """Computes contingency tables for every unique decision value.

    :param ys: true labels
    :type ys: iterable
    :param decision_values: decision values (higher = stronger positive)
    :type decision_values: iterable
    :param positive: the positive label
    :param presorted: whether or not ys and yhat are already sorted
    :type presorted: bool

    :returns: a list of contingency tables `(TP, FP, TN, FN)` and the corresponding thresholds.
    Contingency tables are built based on decision :math:`decision\_value \geq threshold`.

    The first contingency table corresponds with a (potentially unseen) threshold that yields all negatives.

    >>> y = [0, 0, 0, 0, 1, 1, 1, 1]
    >>> d = [2, 2, 1, 1, 1, 2, 3, 3]
    >>> tables, thresholds = contingency_tables(y, d, 1)
    >>> print(tables)
    [(0, 0, 4, 4), (2, 0, 4, 2), (3, 2, 2, 1), (4, 4, 0, 0)]
    >>> print(thresholds)
    [None, 3, 2, 1]

    """
    if presorted:
        if decision_values[0] > decision_values[-1]:
            ind = range(len(decision_values))
            srt = decision_values
        else:
            ind = reversed(range(len(decision_values)))
            srt = reversed(decision_values)
    else:
        # sort decision values
        ind, srt = zip(*sorted(enumerate(decision_values), reverse=True,
                            key=op.itemgetter(1)))

    # resort y
    y = list(map(lambda x: ys[x] == positive, ind))

    num_instances = len(ind)
    total_num_pos = sum(y)

    thresholds = [None]
    tables = [(0, 0, num_instances - total_num_pos, total_num_pos)]

    current_idx = 0
    while current_idx < num_instances:
        # determine number of identical decision values
        num_ties = 1
        while current_idx + num_ties < num_instances and srt[current_idx + num_ties] == srt[current_idx]:
            num_ties += 1

        if current_idx == 0:
            previous_table = (0, 0, num_instances - total_num_pos, total_num_pos)

        # find number of new true positives at this threshold
        num_pos = 0
        for i in range(current_idx, current_idx + num_ties):
            num_pos += y[i]

        # difference compared to previous contingency_table
        diff = (num_pos, num_ties - num_pos, - num_ties + num_pos, -num_pos)

        new_table = tuple(map(op.add, previous_table, diff))
        tables.append(new_table[:])
        thresholds.append(srt[current_idx])

        # prepare for next iteration
        previous_table = new_table
        current_idx += num_ties

    return tables, thresholds


def compute_curve(ys, decision_values, xfun, yfun, positive=True, presorted=False):
    """Computes a curve based on contingency tables at different decision values.

    :param ys: true labels
    :type ys: iterable
    :param decision_values: decision values
    :type decision_values: iterable
    :param positive: positive label
    :param xfun: function to compute x values, based on contingency tables
    :type xfun: callable
    :param yfun: function to compute y values, based on contingency tables
    :type yfun: callable
    :param presorted: whether or not ys and yhat are already sorted
    :type presorted: bool

    :returns: the resulting curve, as a list of (x, y)-tuples

    """
    curve = []
    tables, _ = contingency_tables(ys, decision_values, positive, presorted)
    curve = list(map(lambda t: (xfun(t), yfun(t)), tables))
    return curve


def auc(curve):
    """Computes the area under the specified curve.

    :param curve: a curve, specified as a list of (x, y) tuples
    :type curve: [(x, y), ...]

    .. seealso:: :func:`optunity.score_functions.compute_curve`

    """
    area = 0.0
    for i in range(len(curve) - 1):
        x1, y1 = curve[i]
        x2, y2 = curve[i + 1]
        if y1 is None:
            y1 = 0.0
        area += float(min(y1, y2)) * float(x2 - x1) + math.fabs(float(y2 - y1)) * float(x2 - x1) / 2

    return area


def contingency_table(ys, yhats, positive=True):
    """Computes a contingency table for given predictions.

    :param ys: true labels
    :type ys: iterable
    :param yhats: predicted labels
    :type yhats: iterable
    :param positive: the positive label

    :return: TP, FP, TN, FN

    >>> ys =    [True, True, True, True, True, False]
    >>> yhats = [True, True, False, False, False, True]
    >>> tab = contingency_table(ys, yhats, 1)
    >>> print(tab)
    (2, 1, 0, 3)

    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for y, yhat in zip(ys, yhats):
        if y == positive:
            if y == yhat:
                TP += 1
            else:
                FN += 1
        else:
            if y == yhat:
                TN += 1
            else:
                FP += 1
    return TP, FP, TN, FN

def _precision(table):
    TP = table[0]
    FP = table[1]
    try:
        return float(TP) / (TP + FP)
    except ZeroDivisionError:
        return None

def _recall(table):
    TP = table[0]
    FN = table[3]
    try:
        return float(TP) / (TP + FN)
    except ZeroDivisionError:
        return None

def _fpr(table):
    FP = table[1]
    TN = table[2]
    return float(FP) / (FP + TN)


def mse(y, yhat):
    """Returns the mean squared error between y and yhat.

    :param y: true function values
    :param yhat: predicted function values
    :returns:
        .. math:: \\frac{1}{n} \sum_{i=1}^n \\big[(\hat{y}-y)^2\\big]

    Lower is better.

    >>> mse([0, 0], [2, 3])
    6.5

    """
    return float(sum([(l - p) ** 2
                      for l, p in zip(y, yhat)])) / len(y)


def absolute_error(y, yhat):
    """Returns the maximal absolute error between y and yhat.

    :param y: true function values
    :param yhat: predicted function values

    Lower is better.

    >>> absolute_error([0,1,2,3], [0,0,1,1])
    2.0

    """
    return float(max(map(lambda x, y: math.fabs(x-y), y, yhat)))


def accuracy(y, yhat):
    """Returns the accuracy. Higher is better.

    :param y: true function values
    :param yhat: predicted function values

    """
    return float(sum(map(lambda x: x[0] == x[1],
                            zip(y, yhat)))) / len(y)


def logloss(y, yhat):
    """Returns the log loss between labels and predictions.

    :param y: true function values
    :param yhat: predicted function values
    :returns:
        .. math:: -\\frac{1}{n}\sum_{i=1}^n\\big[y \\times \log \hat{y}+(1-y) \\times \log (1-\hat{y})\\big]

    y must be a binary vector, e.g. elements in {True, False}
    yhat must be a vector of probabilities, e.g. elements in [0, 1]

    Lower is better.

    .. note:: This loss function should only be used for probabilistic models.

    """
    loss = sum([math.log(pred) for _, pred in
                filter(lambda i: i[0], zip(y, yhat))])
    loss += sum([math.log(1 - pred) for _, pred in
                filter(lambda i: not i[0], zip(y, yhat))])
    return - float(loss) / len(y)


def brier(y, yhat, positive=True):
    """Returns the Brier score between y and yhat.

    :param y: true function values
    :param yhat: predicted function values
    :returns:
        .. math:: \\frac{1}{n} \sum_{i=1}^n \\big[(\hat{y}-y)^2\\big]

    yhat must be a vector of probabilities, e.g. elements in [0, 1]

    Lower is better.

    .. note:: This loss function should only be used for probabilistic models.

    """
    y = map(lambda x: x == positive, y)
    return sum([(yp - float(yt)) ** 2 for yt, yp in zip(y, yhat)]) / len(y)


def pu_score(y, yhat):
    """
    Returns a score used for PU learning as introduced in [LEE2003]_.

    :param y: true function values
    :param yhat: predicted function values
    :returns:
        .. math:: \\frac{P(\hat{y}=1 | y=1)^2}{P(\hat{y}=1)}

    y and yhat must be boolean vectors.

    Higher is better.

    .. [LEE2003] Wee Sun Lee and Bing Liu. Learning with positive and unlabeled examples
        using weighted logistic regression. In Proceedings of the Twentieth
        International Conference on Machine Learning (ICML), 2003.
    """
    num_pos = sum(y)
    p_pred_pos = float(sum(yhat)) / len(y)
    if p_pred_pos == 0:
        return 0.0
    tp = sum([all(x) for x in zip(y, yhat)])
    return tp * tp / (num_pos * num_pos * p_pred_pos)

def fbeta(y, yhat, beta, positive=True):
    """Returns the :math:`F_\\beta`-score.

    :param y: true function values
    :param yhat: predicted function values
    :param beta: the value for beta to be used
    :type beta: float (positive)
    :param positive: the positive label

    :returns:
        .. math:: (1 + \\beta^2)\\frac{cdot precision\\cdot recall}{(\\beta^2 * precision)+recall}

    """
    bsq = beta ** 2
    TP, FP, _, FN = contingency_table(y, yhat, positive)
    return float(1 + bsq) * TP / ((1 + bsq) * TP + bsq * FN + FP)

def precision(y, yhat, positive=True):
    """Returns the precision (higher is better).

    :param y: true function values
    :param yhat: predicted function values
    :param positive: the positive label

    :returns: number of true positive predictions / number of positive predictions

    """
    TP, FP, _, _ = contingency_table(y, yhat, positive)
    return _precision(TP, FP)

def recall(y, yhat, positive=True):
    """Returns the recall (higher is better).

    :param y: true function values
    :param yhat: predicted function values
    :param positive: the positive label

    :returns: number of true positive predictions / number of true positives

    """
    TP, _, _, FN = contingency_table(y, yhat, positive)
    return _recall(TP, FN)

def npv(y, yhat, positive=True):
    """Returns the negative predictive value (higher is better).

    :param y: true function values
    :param yhat: predicted function values
    :param positive: the positive label

    :returns: number of true negative predictions / number of negative predictions

    """
    _, _, TN, FN = contingency_table(y, yhat, positive)
    return float(TN) / (TN + FN)

def error_rate(y, yhat):
    """Returns the error rate (lower is better).

    :param y: true function values
    :param yhat: predicted function values

    >>> error_rate([0,0,1,1], [0,0,0,1])
    0.25

    """
    return 1.0 - accuracy(y, yhat)

def roc_auc(ys, yhat, positive=True, presorted=False, return_curve=False):
    """Computes the area under the receiver operating characteristic curve (higher is better).

    :param y: true function values
    :param yhat: predicted function values
    :param positive: the positive label
    :param presorted: whether or not ys and yhat are already sorted
    :type presorted: bool
    :param return_curve: whether or not the curve should be returned
    :type return_curve: bool

    >>> roc_auc([0, 0, 1, 1], [0, 0, 1, 1], 1)
    1.0

    >>> roc_auc([0,0,1,1], [0,1,1,2], 1)
    0.875

    """
    curve = compute_curve(ys, yhat, _fpr, _recall, positive)
    if return_curve:
        return auc(curve), curve
    else:
        return auc(curve)


def pr_auc(ys, yhat, positive=True, presorted=False, return_curve=False):
    """Computes the area under the precision-recall curve (higher is better).

    :param y: true function values
    :param yhat: predicted function values
    :param positive: the positive label
    :param presorted: whether or not ys and yhat are already sorted
    :type presorted: bool
    :param return_curve: whether or not the curve should be returned
    :type return_curve: bool

    >>> pr_auc([0, 0, 1, 1], [0, 0, 1, 1], 1)
    1.0

    >>> round(pr_auc([0,0,1,1], [0,1,1,2], 1), 2)
    0.92

    .. note:: Precision is undefined at recall = 0.
        In this case, we set precision equal to the precision that was obtained at the lowest non-zero recall.

    """
    curve = compute_curve(ys, yhat, _recall, _precision, positive, presorted)
    # precision is undefined when no positives are predicted
    # we approximate by using the precision at the lowest recall
    curve[0] = (0.0, curve[1][1])
    if return_curve:
        return auc(curve), curve
    else:
        return auc(curve)

def r_squared(y, yhat):
    """Returns the R squared statistic, also known as coefficient of determination (higher is better).

    :param y: true function values
    :param yhat: predicted function values
    :param positive: the positive label

    :returns:
        .. math:: R^2 = 1-\\frac{SS_{res}}{SS_{tot}} = 1-\\frac{\sum_i (y_i - yhat_i)^2}{\sum_i (y_i - mean(y))^2}

    """
    ymean = float(sum(y)) / len(y)
    SStot = sum(map(lambda yi: (yi-ymean) ** 2, y))
    SSres = sum(map(lambda yi, fi: (yi-fi) ** 2, y, yhat))
    return 1.0 - SSres / SStot
