# IPYTHON=1 .bin/pyshark

from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array

import optunity

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("sample_svm_data.txt")
parsedData = data.map(parsePoint).cache()

# cross-validation using optunity
@optunity.cross_validated(x=parsedData, num_folds=5, num_iter=1)
def logistic_l2_accuracy(x_train, x_test, regParam):
    # cache data to get reasonable speeds for methods like LogisticRegression and SVM
    xc = x_train.cache()
    # training logistic regression with L2 regularization
    model = LogisticRegressionWithSGD.train(xc, regParam=regParam, regType="l2")
    # making prediction on x_test
    yhat  = x_test.map(lambda p: (p.label, model.predict(p.features)))
    # compute test accuracy
    return yhat.filter(lambda (v, p): v == p).count() / float(x_test.count())

optimal_pars, _, _ = optunity.maximize(logistic_l2_accuracy, num_evals=10, regParam=[0, 10])

# training model with found parameters using all data
model = LogisticRegressionWithSGD.train(parsedData, regType="l2", **optimal_pars)


