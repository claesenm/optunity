SVM (e1071)
============================================

.. include:: /global.rst
.. highlight:: r

In this example, we will train an SVM with RBF kernel and tune its hyperparameters, i.e. **cost** and **gamma** in log-scale.
We will use particle swarms to maximize AUC of Precision-Recall (**auc_pr**) of twice iterated 5-fold cross-validation::

    library(optunity)
    library(e1071)

    ## artificial data
    N <- 100
    x <- matrix(runif(N*5), N, 5)
    y <- x[,1] + 0.5*x[,2]^2 + 0.1*runif(N) + sin(x[,3]) > 1.0
    
    ## SVM train and predict
    svm.rbf <- function(x, y, xtest, ytest, logCost, logGamma) {
        m <- svm(x, y, kernel="radial", cost=exp(logCost), gamma=exp(logGamma), 
                 type="C-classification", probability=TRUE)
        yscore <- attr(predict(m, xtest, probability= T), "probabilities")[,"TRUE"]
        return(yscore)
    }

    cv <- cv.setup(x, y, score=auc_pr, num_folds = 5, num_iter = 2)
    res <- cv.particle_swarm(cv, svm.rbf, logCost = c(-5, 5), logGamma = c(-5, 5), maximize = TRUE)

    ## optimal value for logC:
    res$solution

    ## auc_pr reached inside CV
    res$optimum

Here we used default settings for **cv.particle_swarm**, see `?cv.particle_swarm` in R for details.
