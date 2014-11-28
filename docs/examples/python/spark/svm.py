# IPYTHON=1 .bin/pyshark

from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("sample_svm_data.txt")
parsedData = data.map(parsePoint)

# Build the model
model = LogisticRegressionWithSGD.train(parsedData)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

# cross-validation using optunity
@optunity.cross_validated(x=parsedData, num_folds=5, num_iter=2)
def logistic_l2_accuracy(x_train, x_test, regParam):
    # training logistic regression with L2 regularization
    model = LogisticRegressionWithSGD.train(x_train, regParam=regParam, regType="l2")
    # making prediction on x_test
    yhat  = x_test.map(lambda p: (p.label, model.predict(p.features)))
    # compute test accuracy
    return yhat.filter(lambda (v, p): v == p).count() / float(x_test.count())

optimal_pars, _, _ = optunity.maximize(logistic_l2_accuracy, num_evals=200, regParam=[0, 10])

