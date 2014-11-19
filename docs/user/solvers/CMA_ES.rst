CMA-ES
=======

.. include:: /global.rst

This solver is implemented in |api-cmaes|. It as available in |make_solver| as 'cma-es'.

CMA-ES stands for Covariance Matrix Adaptation Evolutionary Strategy. This is an evolutionary strategy for continuous function optimization. It can dynamically
adapt its search resolution per hyperparameter, allowing for efficient searches on different scales. More information is available in [HANSEN2001]_.

Optunity's implementation of this solver is done using the DEAP toolbox [DEAP2012]_. This, in turn, requires NumPy. Both dependencies must be met to use this solver.

Bibliographic references:

.. [HANSEN2001] Nikolaus Hansen and Andreas Ostermeier. *Completely
    derandomized self-adaptation in evolution  strategies*.
    Evolutionary computation, 9(2):159-195, 2001.

.. [DEAP2012] Felix-Antoine Fortin, Francois-Michel De Rainville, Marc-Andre Gardner,
    Marc Parizeau and Christian Gagne, *DEAP: Evolutionary Algorithms Made Easy*,
    Journal of Machine Learning Research, pp. 2171-2175, no 13, jul 2012.
