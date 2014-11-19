Particle Swarm Optimization
============================

.. include:: /global.rst

This solver is implemented in |api-pso|. It as available in |make_solver| as 'particle swarm'.


Particle swarm optimization (PSO) is a heuristic optimization technique. It simulates a set of particles (candidate solutions)
that are moving aroud in the search-space [PSO2010]_, [PSO2002]_. 

In the context of hyperparameter search, the position of a particle represents a set of hyperparameters and its movement is
influenced by the goodness of the objective function value.

PSO is an iterative algorithm::

1. **Initialization**: a set of particles is initialized with random positions and initial velocities. The initialization step
   is essentially equivalent to :doc:`/user/solvers/random_search`.

2. **Iteration**: every particle's position is updated based on its velocity, the particle's historically best position and
   the entire swarm's historical optimum. 

    .. figure:: pso_iteration.png
        :alt: PSO iterations

        Particle swarm iterations:

        -   :math:`\vec{x}_i` is a particle's position,
        -   :math:`\vec{v}_i` its velocity,
        -   :math:`\vec{p}_i` its historically best position,
        -   :math:`\vec{p}_g` is the swarm's optimum,
        -   :math:`\vec{\phi}_1` and :math:`\vec{\phi}_2` are vectors of uniformly sampled values in :math:`(0, \phi_1)` and :math:`(0, \phi_2)`, respectively.

PSO has 5 parameters that can be configured (see :py:class:`optunity.solvers.ParticleSwarm`):

-   `num_particles`: the number of particles to use
-   `num_generations`: the number of generations (iterations)
-   `phi1`: the impact of each particle's historical best on its movement
-   `phi2`: the impact of the swarm's optimum on the movement of each particle
-   `max_speed`: an upper bound for :math:`\vec{v}_i`

The number of function evaluations that will be performed is `num_particles * num_generations`. A high number of particles
focuses on global, undirected search (just like :doc:`/user/solvers/random_search`), whereas a high number of generations
leads to more localized search since all particles will have time to converge.

Bibliographic references:

.. [PSO2010] Kennedy, James. *Particle swarm optimization*. Encyclopedia of Machine Learning. Springer US, 2010. 760-766.

.. [PSO2002] Clerc, Maurice, and James Kennedy. *The particle swarm-explosion, stability, and convergence in a multidimensional complex space*. 
    Evolutionary Computation, IEEE Transactions on 6.1 (2002): 58-73.
