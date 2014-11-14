import optunity

def f(x, y):
    return -x**2 - y**2

optimal_pars, details, _ = optunity.maximize(f, num_evals=200, x=[-5, 5], y=[-5, 5])

print('optimal parameters: ' + str(optimal_pars))
print('optimal value (true optimum=0): ' + str(details.optimum))
