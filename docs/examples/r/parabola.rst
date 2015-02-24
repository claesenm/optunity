Optimizing a simple 2D parabola
================================

.. include:: /global.rst
.. highlight:: r

In this example, we will use Optunity in R to maximize a very simple function, namely a two-dimensional parabola.

More specifically, the objective function is :math:`f(x, y) = -x^2 - y^2`.

The full code in R::

    library(optunity)

    f   <- function(x,y) -x^2 - y^2
    opt <- particle_swarm(f, x=c(-5, 5), y=c(-5, 5) )
  
In this simple example we used particle swarms optimization with default settings (number of particles 5, number of generations 10).

An example with 10 particles and 15 generations::

    opt <- particle_swarm(f, x=c(-5, 5), y=c(-5, 5), num_particles=10, num_generations=15)

In addition to `particle_swarm` the R interface has `grid_search`, `random_search`, `nelder_mead`. For examples with them use R's internal help, e.g. `?random_search`.
