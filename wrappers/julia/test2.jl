using Base.Test

include("optunity.jl")

testit_dict(d::Dict) = -(d[:x]-1)^2 - (d[:y]-2)^2 - (d[:z]+3)^4

vars, details = maximize(testit_dict, num_evals=10000, solver_name="grid search", x=[-5,5], y=[-5,5], z=[-5,5])

@test_approx_eq_eps vars["x"]  1.0 .2
@test_approx_eq_eps vars["y"]  2.0 .2
@test_approx_eq_eps vars["z"] -3.0 .2

vars, details = maximize(testit_dict, num_evals=10000, z=[-5,5], y=[-5,5], x=[-5,5])

@test_approx_eq_eps vars["x"]  1.0 .2
@test_approx_eq_eps vars["y"]  2.0 .2
@test_approx_eq_eps vars["z"] -3.0 .2

vars, details = maximize(testit_dict, num_evals=10000, solver_name="nelder-mead", x=[-5,5], y=[-5,5], z=[-5,5])

@test_approx_eq_eps vars["x"]  0.8 .1
@test_approx_eq_eps vars["y"]  1.9 .1
@test_approx_eq_eps vars["z"] -3.4 .1