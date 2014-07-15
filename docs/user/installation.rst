====================
Installing Optunity
====================

.. toctree::
    :hidden:

    /wrappers/matlab/installation
    /wrappers/R/installation

Optunity can be installed as a typical Python package.

.. note::

    Optunity has soft dependencies on NumPy_ and SciPy_ 
    (for the :doc:`CMA-ES </user/solvers/CMA_ES>` and 
    :doc:`Nelder-Mead </user/solvers/nelder-mead>` solvers, respectively).
    Optunity additionally ships with DEAP, a library for evolutionary algorithms [DEAP2012]_.

    .. [DEAP2012] Fortin, Félix-Antoine, et al. "DEAP: Evolutionary algorithms made easy."
        Journal of Machine Learning Research 13.1 (2012): 2171-2175.

    .. _NumPy:
        http://www.numpy.org

    .. _SciPy:
        http://www.scipy.org

Setting up Optunity for other environments
===========================================

If you want to use Optunity in a non-Python environment, please refer to the environment-specific installation instructions.

- :doc:`MATLAB installation <../wrappers/matlab/installation>`
- :doc:`R installation <../wrappers/R/installation>`
