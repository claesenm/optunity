Solver Overview
===================

.. include:: /global.rst

The following solvers are available in Optunity:

.. toctree::
    :maxdepth: 2
    :glob:

    /user/solvers/grid_search
    /user/solvers/random_search
    /user/solvers/particle_swarm
    /user/solvers/nelder-mead
    /user/solvers/CMA_ES

You can specify which solver you want to use in |maximize| and |minimize|, but only limited configuration is possible.
If you want to specify detailed settings for each solver, you can use the expert interface, specifically |make_solver| in combination with |optimize|.

We currently recommend the |pso| solver (our default). Based on our experience this is the most reliable solver across different learning algorithms. 
If you consistently get great results with a solver/algorithm combination, we are happy to hear about your experiences.
