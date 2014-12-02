#!/usr/bin/env python

# A simple smoke test for all available solvers.

import optunity

def f(x, y):
    return x + y

solvers = optunity.available_solvers()

for solver in solvers:
    # simple API
    opt, _, _ = optunity.maximize(f, 100,
                                  x=[0, 5], y=[-5, 5],
                                  solver_name=solver)

    # expert API
    suggestion = optunity.suggest_solver(num_evals=100, x=[0, 5], y=[-5, 5],
                                         solver_name=solver)
    s = optunity.make_solver(**suggestion)
    # without parallel evaluations
    opt, _ = optunity.optimize(s, f)
    # with parallel evaluations
    opt, _ = optunity.optimize(s, f, pmap=optunity.pmap)
