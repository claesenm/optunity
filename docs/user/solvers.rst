Solver overview
===================

.. include:: /global.rst

We currently recommend using |pso| (our default). Based on our experience this is the most reliable solver across different learning algorithms. 
If you consistently get great results with a solver/algorithm combination, we are happy to hear about your experiences.

You can specify which solver you want to use in |maximize| and |minimize|, but only limited configuration is possible.
If you want to specify detailed settings for a solver, you can use the expert interface, specifically |make_solver| in combination with |optimize|.

The following solvers are available in Optunity:

.. toctree::
    :maxdepth: 1
    :glob:

    /user/solvers/grid_search
    /user/solvers/random_search
    /user/solvers/particle_swarm
    /user/solvers/nelder-mead
    /user/solvers/CMA_ES
    /user/solvers/TPE
    /user/solvers/sobol

Optunity's default solver is |pso|.

|gridsearch|, |randomsearch| and |sobol| are completely undirected algorithms and consequently not very efficient. 
Of these three, |sobol| is most efficient as uses a low-discrepancy quasirandom sequence. 

|nelder-mead| works well for objective functions that are smooth, unimodal and not too noisy (it is good for local search when you have a good idea about optimal regions for your hyperparameters). 

For general searches, |pso| and |cmaes| are most robust. Finally, the |tpe| solver exposes Hyperopt's TPE solver in Optunity's familiar API.
