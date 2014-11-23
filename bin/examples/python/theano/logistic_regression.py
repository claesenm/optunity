import numpy
import theano
import theano.tensor as T
from numpy.random import multivariate_normal
rng = numpy.random

import optunity
import optunity.metrics

########################
# CREATE THE DATA SET
########################

N = 200
feats = 2
noise_level = 1
data = multivariate_normal((0.0, 0.0), numpy.array([[1.0, 0.0], [0.0, 1.0]]), N)
noise = noise_level * numpy.random.randn(N)
targets = 1 + 2 * data[:,0] + 3 * data[:,1] + noise

median_target = numpy.median(targets)
labels = numpy.array(map(lambda t: 1 if t > median_target else 0, targets))

####################################
# LR regression training function
####################################

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

#############################
# LR evaluation functions
#############################

def lr_untuned(x_train, y_train, x_test, y_test):
    predict, w, b = train_lr(x_train, y_train)
    yhat = predict(x_test)
    loss = optunity.metrics.logloss(y_test, yhat)
    brier = optunity.metrics.brier(y_test, yhat)
    print('+ model: ' + str(b.get_value())[:5] + ' + ' + str(w.get_value()[0])[:5] + ' * x1 + ' + str(w.get_value()[1])[:5] + ' * x2')
    print('++ log loss in test fold: ' + str(loss))
    print('++ Brier loss in test fold: ' + str(brier))
    print('')
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
    print('+ model: ' + str(b.get_value())[:5] + ' + ' + str(w.get_value()[0])[:5] + ' * x1 + ' + str(w.get_value()[1])[:5] + ' * x2')
    print('++ log loss in test fold: ' + str(loss))
    print('++ Brier loss in test fold: ' + str(brier))
    print('')
    return loss, brier

# wrap both evaluation functions in cross-validation
# we will compute two metrics using nested cross-validation
# for this purpose we use list_mean() as aggregator
outer_cv = optunity.cross_validated(x=data, y=labels, num_folds=3,
                                    aggregator=optunity.cross_validation.list_mean)
lr_untuned = outer_cv(lr_untuned)
lr_tuned = outer_cv(lr_tuned)

print('true model: 1 + 2 * x1 + 3 * x2')
print('')

# perform experiment
print('evaluating untuned LR model')
untuned_loss, untuned_brier = lr_untuned()

print('evaluating tuned LR model')
tuned_loss, tuned_brier = lr_tuned()

print('Log loss (lower is better):')
print('untuned: ' + str(untuned_loss))
print('tuned: ' + str(tuned_loss))

print('')
print('Brier loss (lower is better):')
print('untuned: ' + str(untuned_brier))
print('tuned: ' + str(tuned_brier))
