# In this example we will show the difference between a 2-d Sobol sequence
# and sampling uniformly at random in 2 dimensions.
# The Sobol sequence has far lower discrepancy, i.e., the generated samples
# are spread out better in the sampling space.
#
# This example requires matplotlib to generate figures.

import matplotlib.pyplot as plt
import optunity
import random

num_pts = 200   # the number of points to generate
skip = 5000     # the number of initial points of the Sobol sequence to skip

# generate Sobol sequence
res = optunity.solvers.Sobol.i4_sobol_generate(2, num_pts, skip)
x1_sobol, x2_sobol = zip(*res)

# generate uniform points
x1_random = [random.random() for _ in range(num_pts)]
x2_random = [random.random() for _ in range(num_pts)]

# plot results
plt.figure(1)
plt.plot(x1_sobol, x2_sobol, 'o')
plt.title('Sobol sequence')
plt.draw()

plt.figure(2)
plt.plot(x1_random, x2_random, 'ro')
plt.title('Uniform random samples')
plt.show()
