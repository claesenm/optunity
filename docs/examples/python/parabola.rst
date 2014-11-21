Optimizing a simple 2D parabola
================================

In this example, we will use Optunity to optimize a very simple function, namely a two-dimensional parabola.

More specifically, the objective function is :math:`f(x, y) = -x^2 - y^2`.

The full code in Python::

    import optunity

    def f(x, y):
        return -x**2 - y**2

    optimal_pars, details, _ = optunity.maximize(f, num_evals=200, x=[-5, 5], y=[-5, 5])

For such simple functions we would use different solvers in practice, but the main idea remains.

To get a basic understanding of the way various solvers in Optunity work, we can optimize this function with all solvers and plot the resulting call logs.
This code for this is available in `bin/examples/python/parabola.py`.
