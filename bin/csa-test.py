import optunity
import optunity.solvers

s = optunity.solvers.CSA(num_processes=5, num_generations=100, T_0=1, Tacc_0=1, x=[0, 1], y=[0, 1])
f = lambda x, y: -x**2-y**2

sol, _ = optunity.optimize(s, f)

print(sol)
