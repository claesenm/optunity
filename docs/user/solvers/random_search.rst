Random Search 
==============

.. include:: /global.rst

This solver is implemented in |api-randomsearch|. It as available in |make_solver| as 'random search'.

This strategy consists of testing a predefined number of randomly sampled hyperparameter tuples.
Sampling is done uniform at random within specified box constraints.

This solver implements the search strategy described in [RAND]_.

.. [RAND] Bergstra, James, and Yoshua Bengio. *Random search for hyper-parameter optimization*. The Journal of Machine Learning Research 13.1 (2012): 281-305

