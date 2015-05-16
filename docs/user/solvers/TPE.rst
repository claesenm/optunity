Tree-structured Parzen Estimator
================================

.. include:: /global.rst

This solver is implemented in |api-tpe|. It as available in |make_solver| as 'TPE'.

The Tree-structured Parzen Estimator (TPE) is a sequential model-based optimization (SMBO) approach.
SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements,
and then subsequently choose new hyperparameters to test based on this model. 

The TPE approach models :math:`P(x|y)` and :math:`P(y)` where x represents hyperparameters and y the associated quality score. 
:math:`P(x|y)` is modeled by transforming the generative process of hyperparameters, replacing the distributions of the configuration prior
with non-parametric densities. In this solver, Optunity only supports uniform priors within given box constraints.
For more exotic search spaces, please refer to [Hyperopt]_. This optimization approach is described in detail in [TPE2011]_ and [TPE2013]_. 

Optunity provides the TPE solver as is implemented in [Hyperopt]_. This solver is only available if Hyperopt is installed, which in turn requires NumPy. Both dependencies must be met to use this solver.

.. [TPE2011] Bergstra, James S., et al. "Algorithms for hyper-parameter optimization." Advances in Neural Information Processing Systems. 2011.

.. [TPE2013] Bergstra, James, Daniel Yamins, and David Cox. "Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures." Proceedings of The 30th International Conference on Machine Learning. 2013.

.. [Hyperopt] http://jaberg.github.io/hyperopt/
