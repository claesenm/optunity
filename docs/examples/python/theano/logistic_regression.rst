Logistic regression
=====================

.. include:: /global.rst

In this example we will use Theano to train logistic regression models on a simple two-dimensional data set.
We will use Optunity to tune the degree of regularization and step sizes (learning rate). This example requires Theano and NumPy.

We start with the necessary imports::

    import numpy
    from numpy.random import multivariate_normal
    rng = numpy.random

    import theano
    import theano.tensor as T

    import optunity
    import optunity.metrics

The next step is defining our data set. We will generate a random 2-dimensional data set. The generative procedure for the targets is as follows:
:math:`1 + 2 * x_1 + 3 * x_2` + a noise term. We assign binary class labels based on whether or not the target value is higher than the mean target::

    N = 200
    feats = 2
    noise_level = 1
    data = multivariate_normal((0.0, 0.0), numpy.array([[1.0, 0.0], [0.0, 1.0]]), N)
    noise = noise_level * numpy.random.randn(N)
    targets = 1 + 2 * data[:,0] + 3 * data[:,1] + noise

    median_target = numpy.median(targets)
    labels = numpy.array(map(lambda t: 1 if t > median_target else 0, targets))

The next thing we need is a training function for LR models, based on Theano's example_::

    training_steps = 2000

    def train_lr(x_train, y_train, regularization=0.01, step=0.1):
        x = T.matrix("x")
        y = T.vector("y")
        w = theano.shared(rng.randn(feats), name="w")
        b = theano.shared(0., name="b")

        # Construct Theano expression graph
        p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))                 # Probability that target = 1
        prediction = p_1
        xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)           # Cross-entropy loss function
        cost = xent.mean() + regularization * (w ** 2).sum()    # The cost to minimize
        gw, gb = T.grad(cost, [w, b])                           # Compute the gradient of the cost
                                                                # (we shall return to this in a
                                                                # following section of this tutorial)

        # Compile
        train = theano.function(
                inputs=[x,y],
                outputs=[prediction, xent],
                updates=((w, w - step * gw), (b, b - step * gb)))
        predict = theano.function(inputs=[x], outputs=prediction)

        # Train
        for i in range(training_steps):
            train(x_train, y_train)
        return predict, w, b

.. _example: http://deeplearning.net/software/theano/tutorial/examples.html#a-real-example-logistic-regression

Now that we know how to train, we can define a modeling strategy with default and tuned hyperparameters::


    def lr_untuned(x_train, y_train, x_test, y_test):
        predict, w, b = train_lr(x_train, y_train)
        yhat = predict(x_test)
        loss = optunity.metrics.logloss(y_test, yhat)
        brier = optunity.metrics.brier(y_test, yhat)
        return loss, brier

    def lr_tuned(x_train, y_train, x_test, y_test):
        @optunity.cross_validated(x=x_train, y=y_train, num_folds=3)
        def inner_cv(x_train, y_train, x_test, y_test, regularization, step):
            predict, _, _ = train_lr(x_train, y_train,
                                    regularization=regularization, step=step)
            yhat = predict(x_test)
            return optunity.metrics.logloss(y_test, yhat)

        pars, _, _ = optunity.minimize(inner_cv, num_evals=50,
                                    regularization=[0.001, 0.05],
                                    step=[0.01, 0.2])
        predict, w, b = train_lr(x_train, y_train, **pars)
        yhat = predict(x_test)
        loss = optunity.metrics.logloss(y_test, yhat)
        brier = optunity.metrics.brier(y_test, yhat)
        return loss, brier

Note that both modeling functions (train, predict, score) return two score measures (log loss and Brier score). We will evaluate
both modeling approaches using cross-validation and report both performance measures (see |cross_validation|). The cross-validation
decorator::

    outer_cv = optunity.cross_validated(x=data, y=labels, num_folds=3,
                                        aggregator=optunity.cross_validation.list_mean)
    lr_untuned = outer_cv(lr_untuned)
    lr_tuned = outer_cv(lr_tuned)

At this point, `lr_untuned` and `lr_tuned` will return a 3-fold cross-validation estimate of `[logloss, Brier]` when evaluated.


Full code
------------

This example is available in detail in `<optunity>/bin/examples/python/theano/logistic_regression.py`.
Typical output of this script will look like::

    true model: 1 + 2 * x1 + 3 * x2

    evaluating untuned LR model
    + model: -0.18 + 1.679 * x1 + 2.045 * x2
    ++ log loss in test fold: 0.08921125198
    ++ Brier loss in test fold: 0.0786225946458

    + model: -0.36 + 1.449 * x1 + 2.247 * x2
    ++ log loss in test fold: 0.08217097905
    ++ Brier loss in test fold: 0.070741583014

    + model: -0.48 + 1.443 * x1 + 2.187 * x2
    ++ log loss in test fold: 0.10545356515
    ++ Brier loss in test fold: 0.0941325050801

    evaluating tuned LR model
    + model: -0.66 + 2.354 * x1 + 3.441 * x2
    ++ log loss in test fold: 0.07508872472
    ++ Brier loss in test fold: 0.0718020866519

    + model: -0.44 + 2.648 * x1 + 3.817 * x2
    ++ log loss in test fold: 0.0718891792875
    ++ Brier loss in test fold: 0.0638209513581

    + model: -0.45 + 2.689 * x1 + 3.858 * x2
    ++ log loss in test fold: 0.06380803593
    ++ Brier loss in test fold: 0.0590374290183

    Log loss (lower is better):
    untuned: 0.0922785987325000
    tuned: 0.070261979980

    Brier loss (lower is better):
    untuned: 0.0811655609133
    tuned: 0.0648868223427
