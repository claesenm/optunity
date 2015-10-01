
Basic: cross-validation
=======================

This notebook explores the main elements of Optunity's cross-validation
facilities, including:

-  standard cross-validation
-  using strata and clusters while constructing folds
-  using different aggregators

We recommend perusing the related documentation for more details.

Nested cross-validation is available as a separate notebook.

.. code:: python

    import optunity
    import optunity.cross_validation

We start by generating some toy data containing 6 instances which we
will partition into folds.

.. code:: python

    data = list(range(6))
    labels = [True] * 3 + [False] * 3

Standard cross-validation 
--------------------------

Each function to be decorated with cross-validation functionality must
accept the following arguments: - x\_train: training data - x\_test:
test data - y\_train: training labels (required only when y is specified
in the cross-validation decorator) - y\_test: test labels (required only
when y is specified in the cross-validation decorator)

These arguments will be set implicitly by the cross-validation decorator
to match the right folds. Any remaining arguments to the decorated
function remain as free parameters that must be set later on.

Lets start with the basics and look at Optunity's cross-validation in
action. We use an objective function that simply prints out the train
and test data in every split to see what's going on.

.. code:: python

    def f(x_train, y_train, x_test, y_test):
        print("")
        print("train data:\t" + str(x_train) + "\t train labels:\t" + str(y_train))
        print("test data:\t" + str(x_test) + "\t test labels:\t" + str(y_test))
        return 0.0

We start with 2 folds, which leads to equally sized train and test
partitions.

.. code:: python

    f_2folds = optunity.cross_validated(x=data, y=labels, num_folds=2)(f)
    print("using 2 folds")
    f_2folds()


.. parsed-literal::

    using 2 folds
    
    train data:	[4, 2, 0]	 train labels:	[False, True, True]
    test data:	[3, 1, 5]	 test labels:	[False, True, False]
    
    train data:	[3, 1, 5]	 train labels:	[False, True, False]
    test data:	[4, 2, 0]	 test labels:	[False, True, True]




.. parsed-literal::

    0.0



.. code:: python

    # f_2folds as defined above would typically be written using decorator syntax as follows
    # we don't do that in these examples so we can reuse the toy objective function
    
    @optunity.cross_validated(x=data, y=labels, num_folds=2)
    def f_2folds(x_train, y_train, x_test, y_test):
        print("")
        print("train data:\t" + str(x_train) + "\t train labels:\t" + str(y_train))
        print("test data:\t" + str(x_test) + "\t test labels:\t" + str(y_test))
        return 0.0

If we use three folds instead of 2, we get 3 iterations in which the
training set is twice the size of the test set.

.. code:: python

    f_3folds = optunity.cross_validated(x=data, y=labels, num_folds=3)(f)
    print("using 3 folds")
    f_3folds()


.. parsed-literal::

    using 3 folds
    
    train data:	[2, 1, 3, 0]	 train labels:	[True, True, False, True]
    test data:	[5, 4]	 test labels:	[False, False]
    
    train data:	[5, 4, 3, 0]	 train labels:	[False, False, False, True]
    test data:	[2, 1]	 test labels:	[True, True]
    
    train data:	[5, 4, 2, 1]	 train labels:	[False, False, True, True]
    test data:	[3, 0]	 test labels:	[False, True]




.. parsed-literal::

    0.0



If we do two iterations of 3-fold cross-validation (denoted by 2x3
fold), two sets of folds are generated and evaluated.

.. code:: python

    f_2x3folds = optunity.cross_validated(x=data, y=labels, num_folds=3, num_iter=2)(f)
    print("using 2x3 folds")
    f_2x3folds()


.. parsed-literal::

    using 2x3 folds
    
    train data:	[4, 1, 5, 3]	 train labels:	[False, True, False, False]
    test data:	[0, 2]	 test labels:	[True, True]
    
    train data:	[0, 2, 5, 3]	 train labels:	[True, True, False, False]
    test data:	[4, 1]	 test labels:	[False, True]
    
    train data:	[0, 2, 4, 1]	 train labels:	[True, True, False, True]
    test data:	[5, 3]	 test labels:	[False, False]
    
    train data:	[0, 2, 1, 4]	 train labels:	[True, True, True, False]
    test data:	[5, 3]	 test labels:	[False, False]
    
    train data:	[5, 3, 1, 4]	 train labels:	[False, False, True, False]
    test data:	[0, 2]	 test labels:	[True, True]
    
    train data:	[5, 3, 0, 2]	 train labels:	[False, False, True, True]
    test data:	[1, 4]	 test labels:	[True, False]




.. parsed-literal::

    0.0



Using strata and clusters
-------------------------

Strata are defined as sets of instances that should be spread out across
folds as much as possible (e.g. stratify patients by age). Clusters are
sets of instances that must be put in a single fold (e.g. cluster
measurements of the same patient).

Optunity allows you to specify strata and/or clusters that must be
accounted for while construct cross-validation folds. Not all instances
have to belong to a stratum or clusters.

Strata
^^^^^^

We start by illustrating strata. Strata are specified as a list of lists
of instances indices. Each list defines one stratum. We will reuse the
toy data and objective function specified above. We will create 2 strata
with 2 instances each. These instances will be spread across folds. We
create two strata: :math:`\{0, 1\}` and :math:`\{2, 3\}`.

.. code:: python

    strata = [[0, 1], [2, 3]]
    f_stratified = optunity.cross_validated(x=data, y=labels, strata=strata, num_folds=3)(f)
    f_stratified()


.. parsed-literal::

    
    train data:	[0, 4, 2, 5]	 train labels:	[True, False, True, False]
    test data:	[1, 3]	 test labels:	[True, False]
    
    train data:	[1, 3, 2, 5]	 train labels:	[True, False, True, False]
    test data:	[0, 4]	 test labels:	[True, False]
    
    train data:	[1, 3, 0, 4]	 train labels:	[True, False, True, False]
    test data:	[2, 5]	 test labels:	[True, False]




.. parsed-literal::

    0.0



Clusters
^^^^^^^^

Clusters work similarly, except that now instances within a cluster are
guaranteed to be placed within a single fold. The way to specify
clusters is identical to strata. We create two clusters:
:math:`\{0, 1\}` and :math:`\{2, 3\}`. These pairs will always occur in
a single fold.

.. code:: python

    clusters = [[0, 1], [2, 3]]
    f_clustered = optunity.cross_validated(x=data, y=labels, clusters=clusters, num_folds=3)(f)
    f_clustered()


.. parsed-literal::

    
    train data:	[0, 1, 2, 3]	 train labels:	[True, True, True, False]
    test data:	[4, 5]	 test labels:	[False, False]
    
    train data:	[4, 5, 2, 3]	 train labels:	[False, False, True, False]
    test data:	[0, 1]	 test labels:	[True, True]
    
    train data:	[4, 5, 0, 1]	 train labels:	[False, False, True, True]
    test data:	[2, 3]	 test labels:	[True, False]




.. parsed-literal::

    0.0



Strata and clusters
^^^^^^^^^^^^^^^^^^^

Strata and clusters can be used together. Lets say we have the following
configuration:

-  1 stratum: :math:`\{0, 1, 2\}`
-  2 clusters: :math:`\{0, 3\}`, :math:`\{4, 5\}`

In this particular example, instances 1 and 2 will inevitably end up in
a single fold, even though they are part of one stratum. This happens
because the total data set has size 6, and 4 instances are already in
clusters.

.. code:: python

    strata = [[0, 1, 2]]
    clusters = [[0, 3], [4, 5]]
    f_strata_clustered = optunity.cross_validated(x=data, y=labels, clusters=clusters, strata=strata, num_folds=3)(f)
    f_strata_clustered()


.. parsed-literal::

    
    train data:	[4, 5, 0, 3]	 train labels:	[False, False, True, False]
    test data:	[1, 2]	 test labels:	[True, True]
    
    train data:	[1, 2, 0, 3]	 train labels:	[True, True, True, False]
    test data:	[4, 5]	 test labels:	[False, False]
    
    train data:	[1, 2, 4, 5]	 train labels:	[True, True, False, False]
    test data:	[0, 3]	 test labels:	[True, False]




.. parsed-literal::

    0.0



Aggregators 
------------

Aggregators are used to combine the scores per fold into a single
result. The default approach used in cross-validation is to take the
mean of all scores. In some cases, we might be interested in worst-case
or best-case performance, the spread, ...

Opunity allows passing a custom callable to be used as aggregator.

The default aggregation in Optunity is to compute the mean across folds.

.. code:: python

    @optunity.cross_validated(x=data, num_folds=3)
    def f(x_train, x_test):
        result = x_test[0]
        print(result)
        return result
    
    f(1)


.. parsed-literal::

    4
    1
    2




.. parsed-literal::

    2.3333333333333335



This can be replaced by any function, e.g. min or max.

.. code:: python

    @optunity.cross_validated(x=data, num_folds=3, aggregator=max)
    def fmax(x_train, x_test):
        result = x_test[0]
        print(result)
        return result
    
    fmax(1)


.. parsed-literal::

    2
    5
    1




.. parsed-literal::

    5



.. code:: python

    @optunity.cross_validated(x=data, num_folds=3, aggregator=min)
    def fmin(x_train, x_test):
        result = x_test[0]
        print(result)
        return result
    
    fmin(1)


.. parsed-literal::

    3
    4
    5




.. parsed-literal::

    3



Retaining intermediate results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often, it may be useful to retain all intermediate results, not just the
final aggregated data. This is made possible via
``optunity.cross_validation.mean_and_list`` aggregator. This aggregator
computes the mean for internal use in cross-validation, but also returns
a list of lists containing the full evaluation results.

.. code:: python

    @optunity.cross_validated(x=data, num_folds=3,
                              aggregator=optunity.cross_validation.mean_and_list)
    def f_full(x_train, x_test, coeff):
        return x_test[0] * coeff
    
    # evaluate f
    mean_score, all_scores = f_full(1.0)
    print(mean_score)
    print(all_scores)



.. parsed-literal::

    2.33333333333
    [0.0, 2.0, 5.0]


Note that a cross-validation based on the ``mean_and_list`` aggregator
essentially returns a tuple of results. If the result is iterable, all
solvers in Optunity use the first element as the objective function
value. You can let the cross-validation procedure return other useful
statistics too, which you can access from the solver trace.

.. code:: python

    opt_coeff, info, _ = optunity.minimize(f_full, coeff=[0, 1], num_evals=10)
    print(opt_coeff)
    print("call log")
    for args, val in zip(info.call_log['args']['coeff'], info.call_log['values']):
        print(str(args) + '\t\t' + str(val))


.. parsed-literal::

    {'coeff': 0.15771484375}
    call log
    0.34521484375		(0.8055013020833334, [0.0, 0.6904296875, 1.72607421875])
    0.47021484375		(1.09716796875, [0.0, 0.9404296875, 2.35107421875])
    0.97021484375		(2.2638346354166665, [0.0, 1.9404296875, 4.85107421875])
    0.72021484375		(1.6805013020833333, [0.0, 1.4404296875, 3.60107421875])
    0.22021484375		(0.5138346354166666, [0.0, 0.4404296875, 1.10107421875])
    0.15771484375		(0.3680013020833333, [0.0, 0.3154296875, 0.78857421875])
    0.65771484375		(1.53466796875, [0.0, 1.3154296875, 3.28857421875])
    0.90771484375		(2.1180013020833335, [0.0, 1.8154296875, 4.53857421875])
    0.40771484375		(0.9513346354166666, [0.0, 0.8154296875, 2.03857421875])
    0.28271484375		(0.65966796875, [0.0, 0.5654296875, 1.41357421875])


Cross-validation with scikit-learn 
-----------------------------------

In this example we will show how to use cross-validation methods that
are provided by scikit-learn in conjunction with Optunity. To do this we
provide Optunity with the folds that scikit-learn produces in a specific
format.

In supervised learning datasets often have unbalanced labels. When
performing cross-validation with unbalanced data it is good practice to
preserve the percentage of samples for each class across folds. To
achieve this label balance we will use StratifiedKFold.

.. code:: python

    data = list(range(20))
    labels = [1 if i%4==0 else 0 for i in range(20)]
    
    @optunity.cross_validated(x=data, y=labels, num_folds=5)
    def unbalanced_folds(x_train, y_train, x_test, y_test):
        print("")
        print("train data:\t" + str(x_train) + "\ntrain labels:\t" + str(y_train)) + '\n'
        print("test data:\t" + str(x_test) + "\ntest labels:\t" + str(y_test)) + '\n'
        return 0.0
    
    unbalanced_folds()


.. parsed-literal::

    
    train data:	[16, 6, 4, 14, 0, 11, 19, 5, 9, 2, 12, 8, 7, 10, 18, 3]
    train labels:	[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    
    test data:	[15, 1, 13, 17]
    test labels:	[0, 0, 0, 0]
    
    
    train data:	[15, 1, 13, 17, 0, 11, 19, 5, 9, 2, 12, 8, 7, 10, 18, 3]
    train labels:	[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    
    test data:	[16, 6, 4, 14]
    test labels:	[1, 0, 1, 0]
    
    
    train data:	[15, 1, 13, 17, 16, 6, 4, 14, 9, 2, 12, 8, 7, 10, 18, 3]
    train labels:	[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    
    test data:	[0, 11, 19, 5]
    test labels:	[1, 0, 0, 0]
    
    
    train data:	[15, 1, 13, 17, 16, 6, 4, 14, 0, 11, 19, 5, 7, 10, 18, 3]
    train labels:	[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    
    test data:	[9, 2, 12, 8]
    test labels:	[0, 0, 1, 1]
    
    
    train data:	[15, 1, 13, 17, 16, 6, 4, 14, 0, 11, 19, 5, 9, 2, 12, 8]
    train labels:	[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1]
    
    test data:	[7, 10, 18, 3]
    test labels:	[0, 0, 0, 0]
    




.. parsed-literal::

    0.0



Notice above how the test label sets have a varying number of postive
samples, some have none, some have one, and some have two.

.. code:: python

    from sklearn.cross_validation import StratifiedKFold
    
    stratified_5folds = StratifiedKFold(labels, n_folds=5)
    folds = [[list(test) for train, test in stratified_5folds]]
    
    @optunity.cross_validated(x=data, y=labels, folds=folds, num_folds=5)
    def balanced_folds(x_train, y_train, x_test, y_test):
        print("")
        print("train data:\t" + str(x_train) + "\ntrain labels:\t" + str(y_train)) + '\n'
        print("test data:\t" + str(x_test) + "\ntest labels:\t" + str(y_test)) + '\n'
        return 0.0
    
    balanced_folds()


.. parsed-literal::

    
    train data:	[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    train labels:	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    
    test data:	[0, 1, 2, 3]
    test labels:	[1, 0, 0, 0]
    
    
    train data:	[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    train labels:	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    
    test data:	[4, 5, 6, 7]
    test labels:	[1, 0, 0, 0]
    
    
    train data:	[0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19]
    train labels:	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    
    test data:	[8, 9, 10, 11]
    test labels:	[1, 0, 0, 0]
    
    
    train data:	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19]
    train labels:	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    
    test data:	[12, 13, 14, 15]
    test labels:	[1, 0, 0, 0]
    
    
    train data:	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    train labels:	[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    
    test data:	[16, 17, 18, 19]
    test labels:	[1, 0, 0, 0]
    




.. parsed-literal::

    0.0



Now all of our train sets have four positive samples and our test sets
have one positive sample.

To use predetermined folds, place a list of the test sample idices into
a list. And then insert that list into another list. Why so many nested
lists? Because you can perform multiple cross-validation runs by setting
num\_iter appropriately and then append num\_iter lists of test samples
to the outer most list. Note that the test samples for a given fold are
the idicies that you provide and then the train samples for that fold
are all of the indices from all other test sets joined together. If not
done carefully this may lead to duplicated samples in a train set and
also samples that fall in both train and test sets of a fold if a
datapoint is in multiple folds' test sets.

.. code:: python

    data = list(range(6))
    labels = [True] * 3 + [False] * 3
    
    fold1 = [[0, 3], [1, 4], [2, 5]]
    fold2 = [[0, 5], [1, 4], [0, 3]] # notice what happens when the indices are not unique
    folds = [fold1, fold2]
    
    @optunity.cross_validated(x=data, y=labels, folds=folds, num_folds=3, num_iter=2)
    def multiple_iters(x_train, y_train, x_test, y_test):
        print("")
        print("train data:\t" + str(x_train) + "\t train labels:\t" + str(y_train))
        print("test data:\t" + str(x_test) + "\t\t test labels:\t" + str(y_test))
        return 0.0
    
    multiple_iters()


.. parsed-literal::

    
    train data:	[1, 4, 2, 5]	 train labels:	[True, False, True, False]
    test data:	[0, 3]		 test labels:	[True, False]
    
    train data:	[0, 3, 2, 5]	 train labels:	[True, False, True, False]
    test data:	[1, 4]		 test labels:	[True, False]
    
    train data:	[0, 3, 1, 4]	 train labels:	[True, False, True, False]
    test data:	[2, 5]		 test labels:	[True, False]
    
    train data:	[1, 4, 0, 3]	 train labels:	[True, False, True, False]
    test data:	[0, 5]		 test labels:	[True, False]
    
    train data:	[0, 5, 0, 3]	 train labels:	[True, False, True, False]
    test data:	[1, 4]		 test labels:	[True, False]
    
    train data:	[0, 5, 1, 4]	 train labels:	[True, False, True, False]
    test data:	[0, 3]		 test labels:	[True, False]




.. parsed-literal::

    0.0



