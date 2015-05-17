import math
import optunity
import random
import numpy as np

# check all available solvers
solvers = optunity.available_solvers()
print('Available solvers: ' + ', '.join(solvers))
logs = {}
optima = dict([(s, []) for s in solvers])

# we run experiments a number of times to estimate each solver's variance
particle_details = None
for i in range(100):
    xoff = random.random()
    yoff = random.random()
    def f(x, y):
        return (x - xoff)**2 + (y - yoff)**2

    for solver in solvers:
        pars, details, _ = optunity.minimize(f, num_evals=100, x=[-5, 5], y=[-5, 5],
                                             solver_name=solver)
        optima[solver].append(details.optimum)
        logs[solver] = np.array([details.call_log['args']['x'],
                                 details.call_log['args']['y']])
        if solver == 'particle swarm': particle_details = details

# plot results
print('plotting results')
colors =  ['r', 'g', 'b', 'y', 'k', 'y', 'r', 'g']
markers = ['x', '+', 'o', 's', 'p', 'x', '+', 'o']

delta = 0.025
x = np.arange(-5.0, 5.0, delta)
y = np.arange(-5.0, 5.0, delta)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.figure(1)
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10, alpha=0.5)
for i, solver in enumerate(solvers):
    print(solver)
    plt.scatter(logs[solver][0,:], logs[solver][1,:], c=colors[i], marker=markers[i], alpha=0.80)

plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.axis('equal')
plt.legend(solvers)
plt.draw()
#plt.savefig('parabola_solver_traces.png', transparant=True)
#plt.clf()

from collections import OrderedDict
log_optima = OrderedDict()
means = OrderedDict()
std = OrderedDict()
for k, v in optima.items():
    log_optima[k] = [-math.log10(val) for val in v]
    means[k] = sum(log_optima[k]) / len(v)
    std[k] = np.std(log_optima[k])

plt.figure(2)
plt.barh(np.arange(len(means)), means.values(), height=0.8, xerr=std.values(), alpha=0.5)
plt.xlabel('number of correct digits')
plt.yticks(np.arange(len(means))+0.4, list(means.keys()))
plt.tight_layout()
plt.show()
plt.savefig('parabola_solver_precision.png', transparant=True)
