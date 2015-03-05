Ridge Regression
============================================

.. include:: /global.rst
.. highlight:: r

In this example, we will train a (linear) ridge regression. In this case, we have to tune one hyperparameters: `logC` (regularization).
We will use twice iterated 5-fold cross-validation to test a hyperparameter and minimize mean squared error (`mean_se`)::

    library(optunity)

    ## artificial data
    N <- 50
    x <- matrix(runif(N*5), N, 5)
    y <- x[,1] + 0.5*x[,2] + 0.1*runif(N)

    ## ridge regression
    regr <- function(x, y, xtest, ytest, logC) {
       ## regularization matrix
       C =  diag(x=exp(logC), ncol(x))
       beta = solve(t(x) %*% x + C, t(x) %*% y)
       xtest %*% beta
    }
    cv <- cv.setup(x, y, score=mean_se, num_folds = 5, num_iter = 2)
    res <- cv.particle_swarm(cv, regr, logC = c(-5, 5), maximize = FALSE)

    ## optimal value for logC:
    res$solution$logC

Here we used default settings for cv.particle_swarm, see `?cv.particle_swarm` in R for details.
