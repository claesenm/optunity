====================
Installing Optunity
====================

.. include:: /global.rst

The source can be obtained from git at http://git.optunity.net (recommended), releases can be obtained from
http://releases.optunity.net. Optunity is compatible with Python 2.7 and 3.x. Note that you must have Python installed
before Optunity can be used (all Linux distributions have this, but Windows requires explicit installation).

Obtaining the latest version of Optunity using git can be done as follows::

    git clone https://github.com/claesenm/optunity.git

Installation instructions per environment:

-   :ref:`install-python`
-   :ref:`install-matlab`
-   :ref:`install-octave`
-   :ref:`install-r`
-   :ref:`install-julia`
-   :ref:`install-java`

If you encounter any difficulties during installation, please open an issue at |issues|.

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

        python optunity/setup.py install [--home=/desired/installation/directory/]

-   Install using pip::

        pip install optunity [-t /desired/installation/directory/]

After these steps, you should be able to import ``optunity`` module::

    python -c 'import optunity'

.. _install-matlab:

Installing Optunity for MATLAB
-------------------------------

To install Optunity for MATLAB, you must add `<optunity>/wrappers/matlab/` and its subdirectories to your MATLAB path.
You can set the path in `Preferences -> Path` or using the following commands::

    addpath(genpath('/path/to/optunity/wrappers/matlab/'));
    savepath

After these steps, you should be able to run the examples in `<optunity>/wrappers/matlab/optunity_example.m`::

    optunity_example

.. warning::

    The MATLAB wrapper requires the entire directory structure of Optunity to remain as is. If you only copy the
    `<optunity>/wrappers/matlab` subdirectory it will not work.

.. _install-octave:

Installing Optunity for Octave
-------------------------------

Optunity requires sockets for communication. In Octave, please install the `sockets` package first (available at Octave-Forge)
and at http://octave.sourceforge.net/sockets/.

To install Optunity for Octave, you must add `<optunity>/wrappers/octave/` and its subdirectories to your Octave path::

    addpath(genpath('/path/to/optunity/wrappers/octave/'));
    savepath

After these steps, you should be able to run the examples in `<optunity>/wrappers/octave/optunity/example/optunity_example.m`::

    optunity_example

.. warning::

    The Octave wrapper requires the entire directory structure of Optunity to remain as is. If you only copy the
    `<optunity>/wrappers/octave` subdirectory it will not work.


.. _install-r:

Installing Optunity for R
--------------------------

First install all dependencies. In R, issue the following command::

    install.packages(c("rjson", "ROCR", "enrichvs", "plyr"))

Subsequently, clone the git repository and then issue the following commands::

    cd optunity/wrappers
    R CMD build R/
    R CMD INSTALL optunity_<version number>.tar.gz


.. _install-julia:

Installing Optunity for Julia
-----------------------------

First install all dependencies. In Julia console issue the following command:

.. code-block:: julia
    
    Pkg.add("PyCall")

After this essential package is installed please issue the following command:

.. code-block:: julia

    include("<optunity>/wrappers/julia/optunity.jl")

where <optunity> stands for the root of the working copy of Optunity. 

.. _install-java:

Installing Optunity for Java
-----------------------------

Optunity is available for Java through Jython (v2.7+). To use Optunity via Jython the Python package must be installed first (see above).
