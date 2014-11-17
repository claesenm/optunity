====================
Installing Optunity
====================

The source can be obtained from git at http://git.optunity.net (recommended), releases can be obtained from
http://releases.optunity.net. Optunity is compatible with Python 2.7 and 3.x. Note that you must have Python installed
before Optunity can be used (all Linux distributions have this, but Windows requires explicit installation).

Installation instructions per environment:

-   :ref:`install-python`
-   :ref:`install-matlab`
-   :ref:`install-r`
-   :ref:`install-java`

.. note::

    Optunity has soft dependencies on NumPy_ and [DEAP2012]_ for the :doc:`CMA-ES </user/solvers/CMA_ES>` solver.
    If these Python libraries are unavailable, the CMA-ES solver will be unavailable.

    .. [DEAP2012] Fortin, Félix-Antoine, et al. "DEAP: Evolutionary algorithms made easy."
        Journal of Machine Learning Research 13.1 (2012): 2171-2175.

    .. _NumPy:
        http://www.numpy.org

.. _install-python:

Installing Optunity for Python
-------------------------------

If you want to use Optunity in another environment, this is not required. 
Optunity can be installed as a typical Python package, for example:

-   Add Optunity's root directory (the one you cloned/extracted) to your ``PYTHONPATH`` environment variable.
    Note that this will only affect your current shell. To make it permanent, add this line to your .bashrc (Linux)
    or adjust the PYTHONPATH variable manually (Windows). 
    
    Extending ``PYTHONPATH`` in Linux (not permanent)::

        export PYTHONPATH=$PYTHONPATH:/path/to/optunity/

    Extending ``PYTHONPATH`` in Windows (not permanent)::
        
        set PYTHONPATH=%PYTHONPATH%;/path/to/optunity

-   Install using Optunity' setup script::

        python setup.py install [--home=/desired/installation/directory/]

-   Install using pip::

        pip install optunity [-t /desired/installation/directory/]

After these steps, you should be able to import ``optunity`` module::

    python -c 'import optunity'

.. _install-matlab:

Installing Optunity for MATLAB
-------------------------------

Installing Optunity for MATLAB is trivial. It only requires you to add `<optunity>/wrappers/matlab/` to your path.

.. _install-r:

Installing Optunity for R
--------------------------

TODO

.. _install-java:

Installing Optunity for Java
-----------------------------

Optunity is available for Java through Jython (v2.7+). To use Optunity via Jython the Python package must be installed first (see above).
