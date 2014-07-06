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


def mse(y, yhat):
    """Returns the mean squared error between y and yhat.

    Lower is better."""
    return float(sum([(l - p) ** 2
                      for l, p in zip(y, yhat)])) / len(y)


def accuracy(y, yhat):
    """Returns the accuracy. Higher is better."""
    return float(len(filter(lambda x: x[0] == x[1],
                            zip(y, yhat)))) / len(y)


def log_loss(y, yhat):
    """Returns the log loss between labels and predictions.

    loss = -(y*log(yhat)+(1-y)*log(1-yhat))

    y must be a binary vector, e.g. elements in {True, False}
    yhat must be a vector of probabilities, e.g. elements in [0, 1]

    Lower is better.
    """
    loss = sum([math.log(pred) for _, pred in
                filter(lambda i: i[0], zip(y, yhat))])
    loss += sum([math.log(1 - pred) for _, pred in
                filter(lambda i: not i[0], zip(y, yhat))])
    return -loss


def brier_score(y, yhat):
    """Returns the Brier score between y and yhat.

    score = 1/len(y) * sum((yhat-y)^2)

    y must be a boolean vector, e.g. elements in {True, False}
    yhat must be a vector of probabilities, e.g. elements in [0, 1]

    Lower is better.
    """
    return sum([(yp - float(yt)) ** 2 for yt, yp in zip(y, yhat)]) / len(y)


def pu_score(y, yhat):
    """
    Returns a score used for PU learning.

    score = recall^2 / probability(yhat = 1)

    y and yhat must be boolean vectors.

    Higher is better.

    Reference:
    Wee Sun Lee and Bing Liu. Learning with positive and unlabeled examples
    using weighted logistic regression. In Proceedings of the Twentieth
    International Conference on Machine Learning (ICML), 2003.
    """
    num_pos = sum(y)
    p_pred_pos = float(sum(yhat)) / len(y)
    if p_pred_pos == 0:
        return 0.0
    tp = sum([all(x) for x in zip(y, yhat)])
    return tp * tp / (num_pos * num_pos * p_pred_pos)

