.. |constraints| replace:: :doc:`/user/constraints`
.. |cross_validation| replace:: :doc:`/user/cross_validation`
.. |solvers| replace:: :doc:`/user/solvers`
.. |score_functions| replace:: :doc:`/user/score_functions`
.. |api| replace:: :doc:`/api/optunity`
.. |examples| replace:: :doc:`/examples/index`

.. |pso| replace:: :doc:`/user/solvers/particle_swarm`
.. |cmaes| replace:: :doc:`/user/solvers/CMA_ES`
.. |gridsearch| replace:: :doc:`/user/solvers/grid_search`
.. |randomsearch| replace:: :doc:`/user/solvers/random_search`
.. |nelder-mead| replace:: :doc:`/user/solvers/nelder-mead`

.. |api-pso| replace:: :class:`optunity.solvers.ParticleSwarm`
.. |api-cmaes| replace:: :class:`optunity.solvers.CMA_ES`
.. |api-gridsearch| replace:: :class:`optunity.solvers.GridSearch`
.. |api-randomsearch| replace:: :class:`optunity.solvers.RandomSearch`
.. |api-nelder-mead| replace:: :class:`optunity.solvers.NelderMead`

.. |warning-unconstrained| replace:: 
        .. warning:: This solver is not explicitly constrained. The box constraints that are given
            are used to initialise the swarm, but particles may leave the specified region during iterations.
            If this is unacceptable, you must manually constrain the domain of the objective function
            prior to using this solver (cfr. :doc:`/user/constraints`).
