# This file shows an example of how to retain all information during cross-validation per fold.
# For this we use the optunity.cross_validation.mean_and_list aggregator.
# For more context, cfr. https://github.com/claesenm/optunity/issues/40

import optunity
import optunity.cross_validation

x = list(range(5))

@optunity.cross_validated(x, num_folds=5,
                          aggregator=optunity.cross_validation.mean_and_list)
def f(x_train, x_test, coeff):
    return x_test[0] * coeff

# evaluate f
foo = f(0.0)
print(foo)

opt_coeff, info, _ = optunity.minimize(f, coeff=[0, 1], num_evals=10)
print(opt_coeff)
print("call log")
for args, val in zip(info.call_log['args']['coeff'], info.call_log['values']):
    print(str(args) + '\t\t' + str(val))
