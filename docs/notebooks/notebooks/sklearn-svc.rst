
sklearn: SVM classification
===========================

In this example we will use Optunity to optimize hyperparameters for a
support vector machine classifier (SVC) in scikit-learn. We will learn a
model to distinguish digits 8 and 9 in the MNIST data set in two
settings

-  tune SVM with RBF kernel
-  tune SVM with RBF, polynomial or linear kernel, that is choose the
   kernel function and its hyperparameters at once

.. code:: python

    import optunity
    import optunity.metrics
    
    # comment this line if you are running the notebook
    import sklearn.svm
    import numpy as np
Create the data set: we use the MNIST data set and will build models to
distinguish digits 8 and 9.

.. code:: python

    from sklearn.datasets import load_digits
    digits = load_digits()
    n = digits.data.shape[0]
    
    positive_digit = 8
    negative_digit = 9
    
    positive_idx = [i for i in range(n) if digits.target[i] == positive_digit]
    negative_idx = [i for i in range(n) if digits.target[i] == negative_digit]
    
    # add some noise to the data to make it a little challenging
    original_data = digits.data[positive_idx + negative_idx, ...]
    data = original_data + 5 * np.random.randn(original_data.shape[0], original_data.shape[1])
    labels = [True] * len(positive_idx) + [False] * len(negative_idx)
First, lets see the performance of an SVC with default hyperparameters.

.. code:: python

    # compute area under ROC curve of default parameters
    @optunity.cross_validated(x=data, y=labels, num_folds=5)
    def svm_default_auroc(x_train, y_train, x_test, y_test):
        model = sklearn.svm.SVC().fit(x_train, y_train)
        decision_values = model.decision_function(x_test)
        auc = optunity.metrics.roc_auc(y_test, decision_values)
        return auc
    
    svm_default_auroc()



.. parsed-literal::

    0.7328666183635757



Tune SVC with RBF kernel 
-------------------------

In order to use Optunity to optimize hyperparameters, we start by
defining the objective function. We will use 5-fold cross-validated area
under the ROC curve. For now, lets restrict ourselves to the RBF kernel
and optimize :math:`C` and :math:`\gamma`.

We start by defining the objective function ``svm_rbf_tuned_auroc()``,
which accepts :math:`C` and :math:`\gamma` as arguments.

.. code:: python

    #we will make the cross-validation decorator once, so we can reuse it later for the other tuning task
    # by reusing the decorator, we get the same folds etc.
    cv_decorator = optunity.cross_validated(x=data, y=labels, num_folds=5)
    
    def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, C, logGamma):
        model = sklearn.svm.SVC(C=C, gamma=10 ** logGamma).fit(x_train, y_train)
        decision_values = model.decision_function(x_test)
        auc = optunity.metrics.roc_auc(y_test, decision_values)
        return auc
    
    svm_rbf_tuned_auroc = cv_decorator(svm_rbf_tuned_auroc)
    # this is equivalent to the more common syntax below
    # @optunity.cross_validated(x=data, y=labels, num_folds=5)
    # def svm_rbf_tuned_auroc...
    
    svm_rbf_tuned_auroc(C=1.0, logGamma=0.0)



.. parsed-literal::

    0.5



Now we can use Optunity to find the hyperparameters that maximize AUROC.

.. code:: python

    optimal_rbf_pars, info, _ = optunity.maximize(svm_rbf_tuned_auroc, num_evals=150, C=[0, 10], logGamma=[-5, 0])
    # when running this outside of IPython we can parallelize via optunity.pmap
    # optimal_rbf_pars, _, _ = optunity.maximize(svm_rbf_tuned_auroc, 150, C=[0, 10], gamma=[0, 0.1], pmap=optunity.pmap)
    
    print("Optimal parameters: " + str(optimal_rbf_pars))
    print("AUROC of tuned SVM with RBF kernel: %1.3f" % info.optimum)

.. parsed-literal::

    Optimal parameters: {'logGamma': -3.0716796875000005, 'C': 3.3025997497032007}
    AUROC of tuned SVM with RBF kernel: 0.987


We can turn the call log into a pandas dataframe to efficiently inspect
the solver trace.

.. code:: python

    import pandas
    df = optunity.call_log2dataframe(info.call_log)
Lets look at the best 20 sets of hyperparameters to make sure the
results are somewhat stable.

.. code:: python

    df.sort('value', ascending=False)[:10]



.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>C</th>
          <th>logGamma</th>
          <th>value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>149</th>
          <td> 3.822811</td>
          <td>-3.074680</td>
          <td> 0.987413</td>
        </tr>
        <tr>
          <th>92 </th>
          <td> 3.302600</td>
          <td>-3.071680</td>
          <td> 0.987413</td>
        </tr>
        <tr>
          <th>145</th>
          <td> 3.259690</td>
          <td>-3.033531</td>
          <td> 0.987252</td>
        </tr>
        <tr>
          <th>14 </th>
          <td> 3.542839</td>
          <td>-3.080013</td>
          <td> 0.987237</td>
        </tr>
        <tr>
          <th>131</th>
          <td> 3.232732</td>
          <td>-3.080968</td>
          <td> 0.987237</td>
        </tr>
        <tr>
          <th>53 </th>
          <td> 7.328411</td>
          <td>-3.103471</td>
          <td> 0.987237</td>
        </tr>
        <tr>
          <th>70 </th>
          <td> 3.632562</td>
          <td>-3.088346</td>
          <td> 0.987237</td>
        </tr>
        <tr>
          <th>146</th>
          <td> 3.067660</td>
          <td>-3.091143</td>
          <td> 0.987237</td>
        </tr>
        <tr>
          <th>124</th>
          <td> 2.566381</td>
          <td>-3.114649</td>
          <td> 0.987237</td>
        </tr>
        <tr>
          <th>100</th>
          <td> 3.340268</td>
          <td>-3.092535</td>
          <td> 0.987237</td>
        </tr>
      </tbody>
    </table>
    </div>



Tune SVC without deciding the kernel in advance 
------------------------------------------------

In the previous part we choose to use an RBF kernel. Even though the RBF
kernel is known to work well for a large variety of problems (and
yielded good accuracy here), our choice was somewhat arbitrary.

We will now use Optunity's conditional hyperparameter optimization
feature to optimize over all kernel functions and their associated
hyperparameters at once. This requires us to define the search space.

.. code:: python

    space = {'kernel': {'linear': {'C': [0, 2]},
                        'rbf': {'logGamma': [-5, 0], 'C': [0, 10]},
                        'poly': {'degree': [2, 5], 'C': [0, 5], 'coef0': [0, 2]}
                        }
             }
We will also have to modify the objective function to cope with
conditional hyperparameters. The reason we need to do this explicitly is
because scikit-learn doesn't like dealing with ``None`` values for
irrelevant hyperparameters (e.g. ``degree`` when using an RBF kernel).
Optunity will set all irrelevant hyperparameters in a given set to
``None``.

.. code:: python

    def train_model(x_train, y_train, kernel, C, logGamma, degree, coef0):
        """A generic SVM training function, with arguments based on the chosen kernel."""
        if kernel == 'linear':
            model = sklearn.svm.SVC(kernel=kernel, C=C)
        elif kernel == 'poly':
            model = sklearn.svm.SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
        elif kernel == 'rbf':
            model = sklearn.svm.SVC(kernel=kernel, C=C, gamma=10 ** logGamma)
        else: 
            raise ArgumentError("Unknown kernel function: %s" % kernel)
        model.fit(x_train, y_train)
        return model
    
    def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='linear', C=0, logGamma=0, degree=0, coef0=0):
        model = train_model(x_train, y_train, kernel, C, logGamma, degree, coef0)
        decision_values = model.decision_function(x_test)
        return optunity.metrics.roc_auc(y_test, decision_values)
    
    svm_tuned_auroc = cv_decorator(svm_tuned_auroc)
Now we are ready to go and optimize both kernel function and associated
hyperparameters!

.. code:: python

    optimal_svm_pars, info, _ = optunity.maximize_structured(svm_tuned_auroc, space, num_evals=150)
    print("Optimal parameters" + str(optimal_svm_pars))
    print("AUROC of tuned SVM: %1.3f" % info.optimum)

.. parsed-literal::

    Optimal parameters{'kernel': 'rbf', 'C': 3.634209495387873, 'coef0': None, 'degree': None, 'logGamma': -3.6018043228483627}
    AUROC of tuned SVM: 0.990


Again, we can have a look at the best sets of hyperparameters based on
the call log.

.. code:: python

    df = optunity.call_log2dataframe(info.call_log)
    df.sort('value', ascending=False)



.. raw:: html

    <div style="max-height:1000px;max-width:1500px;overflow:auto;">
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>C</th>
          <th>coef0</th>
          <th>degree</th>
          <th>kernel</th>
          <th>logGamma</th>
          <th>value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>147</th>
          <td> 3.806445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.594290</td>
          <td> 0.990134</td>
        </tr>
        <tr>
          <th>124</th>
          <td> 3.634209</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.601804</td>
          <td> 0.990134</td>
        </tr>
        <tr>
          <th>144</th>
          <td> 4.350397</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.539531</td>
          <td> 0.990128</td>
        </tr>
        <tr>
          <th>82 </th>
          <td> 5.998112</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.611495</td>
          <td> 0.989975</td>
        </tr>
        <tr>
          <th>75 </th>
          <td> 2.245622</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.392871</td>
          <td> 0.989965</td>
        </tr>
        <tr>
          <th>139</th>
          <td> 4.462613</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.391728</td>
          <td> 0.989965</td>
        </tr>
        <tr>
          <th>111</th>
          <td> 2.832370</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.384538</td>
          <td> 0.989965</td>
        </tr>
        <tr>
          <th>92 </th>
          <td> 5.531445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.378162</td>
          <td> 0.989965</td>
        </tr>
        <tr>
          <th>121</th>
          <td> 3.299037</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.617871</td>
          <td> 0.989818</td>
        </tr>
        <tr>
          <th>99 </th>
          <td> 2.812451</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.547038</td>
          <td> 0.989810</td>
        </tr>
        <tr>
          <th>129</th>
          <td> 4.212451</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.518478</td>
          <td> 0.989809</td>
        </tr>
        <tr>
          <th>135</th>
          <td> 3.921212</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.422389</td>
          <td> 0.989800</td>
        </tr>
        <tr>
          <th>90 </th>
          <td> 3.050174</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.431659</td>
          <td> 0.989800</td>
        </tr>
        <tr>
          <th>103</th>
          <td> 3.181445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.525796</td>
          <td> 0.989650</td>
        </tr>
        <tr>
          <th>93 </th>
          <td> 2.714779</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.292463</td>
          <td> 0.989641</td>
        </tr>
        <tr>
          <th>89 </th>
          <td> 2.345784</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.313704</td>
          <td> 0.989641</td>
        </tr>
        <tr>
          <th>149</th>
          <td> 3.995946</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.303042</td>
          <td> 0.989641</td>
        </tr>
        <tr>
          <th>100</th>
          <td> 3.516840</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.664992</td>
          <td> 0.989500</td>
        </tr>
        <tr>
          <th>119</th>
          <td> 3.745784</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.678403</td>
          <td> 0.989500</td>
        </tr>
        <tr>
          <th>125</th>
          <td> 4.387879</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.486348</td>
          <td> 0.989485</td>
        </tr>
        <tr>
          <th>24 </th>
          <td> 1.914779</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.476204</td>
          <td> 0.989484</td>
        </tr>
        <tr>
          <th>136</th>
          <td> 5.865572</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.226204</td>
          <td> 0.989483</td>
        </tr>
        <tr>
          <th>80 </th>
          <td> 2.583507</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.198326</td>
          <td> 0.989482</td>
        </tr>
        <tr>
          <th>146</th>
          <td> 5.398905</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.459538</td>
          <td> 0.989325</td>
        </tr>
        <tr>
          <th>102</th>
          <td> 5.558878</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.467218</td>
          <td> 0.989325</td>
        </tr>
        <tr>
          <th>108</th>
          <td> 2.721828</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.463704</td>
          <td> 0.989325</td>
        </tr>
        <tr>
          <th>98 </th>
          <td> 2.255162</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.230371</td>
          <td> 0.989324</td>
        </tr>
        <tr>
          <th>64 </th>
          <td> 1.686680</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.240209</td>
          <td> 0.989320</td>
        </tr>
        <tr>
          <th>140</th>
          <td> 3.965939</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.241095</td>
          <td> 0.989320</td>
        </tr>
        <tr>
          <th>34 </th>
          <td> 2.381445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-3.242871</td>
          <td> 0.989320</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>68 </th>
          <td> 1.608145</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.530371</td>
          <td> 0.979475</td>
        </tr>
        <tr>
          <th>106</th>
          <td> 5.681445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.526204</td>
          <td> 0.979156</td>
        </tr>
        <tr>
          <th>50 </th>
          <td> 1.477928</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.498326</td>
          <td> 0.977076</td>
        </tr>
        <tr>
          <th>35 </th>
          <td> 2.081445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.459538</td>
          <td> 0.974526</td>
        </tr>
        <tr>
          <th>15 </th>
          <td> 3.014779</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.459538</td>
          <td> 0.974526</td>
        </tr>
        <tr>
          <th>71 </th>
          <td> 1.464779</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.451204</td>
          <td> 0.973405</td>
        </tr>
        <tr>
          <th>49 </th>
          <td> 2.239779</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.380371</td>
          <td> 0.969723</td>
        </tr>
        <tr>
          <th>9  </th>
          <td> 4.106445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.380371</td>
          <td> 0.969723</td>
        </tr>
        <tr>
          <th>53 </th>
          <td> 3.648112</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.359129</td>
          <td> 0.968756</td>
        </tr>
        <tr>
          <th>17 </th>
          <td> 0.131419</td>
          <td>NaN</td>
          <td>NaN</td>
          <td> linear</td>
          <td>      NaN</td>
          <td> 0.967925</td>
        </tr>
        <tr>
          <th>6  </th>
          <td> 1.913086</td>
          <td>NaN</td>
          <td>NaN</td>
          <td> linear</td>
          <td>      NaN</td>
          <td> 0.967925</td>
        </tr>
        <tr>
          <th>26 </th>
          <td> 1.726419</td>
          <td>NaN</td>
          <td>NaN</td>
          <td> linear</td>
          <td>      NaN</td>
          <td> 0.967925</td>
        </tr>
        <tr>
          <th>7  </th>
          <td> 0.038086</td>
          <td>NaN</td>
          <td>NaN</td>
          <td> linear</td>
          <td>      NaN</td>
          <td> 0.967925</td>
        </tr>
        <tr>
          <th>27 </th>
          <td> 0.224753</td>
          <td>NaN</td>
          <td>NaN</td>
          <td> linear</td>
          <td>      NaN</td>
          <td> 0.967925</td>
        </tr>
        <tr>
          <th>16 </th>
          <td> 1.819753</td>
          <td>NaN</td>
          <td>NaN</td>
          <td> linear</td>
          <td>      NaN</td>
          <td> 0.967925</td>
        </tr>
        <tr>
          <th>37 </th>
          <td> 0.318086</td>
          <td>NaN</td>
          <td>NaN</td>
          <td> linear</td>
          <td>      NaN</td>
          <td> 0.967925</td>
        </tr>
        <tr>
          <th>58 </th>
          <td> 2.074811</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.297038</td>
          <td> 0.964444</td>
        </tr>
        <tr>
          <th>61 </th>
          <td> 1.931445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.217871</td>
          <td> 0.960290</td>
        </tr>
        <tr>
          <th>19 </th>
          <td> 3.639779</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.147038</td>
          <td> 0.958086</td>
        </tr>
        <tr>
          <th>39 </th>
          <td> 2.706445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.147038</td>
          <td> 0.958086</td>
        </tr>
        <tr>
          <th>43 </th>
          <td> 4.114779</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.125796</td>
          <td> 0.957296</td>
        </tr>
        <tr>
          <th>48 </th>
          <td> 2.541478</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-2.063704</td>
          <td> 0.954737</td>
        </tr>
        <tr>
          <th>51 </th>
          <td> 2.398112</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-1.984538</td>
          <td> 0.951944</td>
        </tr>
        <tr>
          <th>29 </th>
          <td> 3.173112</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-1.913704</td>
          <td> 0.942719</td>
        </tr>
        <tr>
          <th>41 </th>
          <td> 2.864779</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-1.751204</td>
          <td> 0.634160</td>
        </tr>
        <tr>
          <th>11 </th>
          <td> 4.264779</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-1.051204</td>
          <td> 0.500000</td>
        </tr>
        <tr>
          <th>31 </th>
          <td> 3.331445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-1.517871</td>
          <td> 0.500000</td>
        </tr>
        <tr>
          <th>1  </th>
          <td> 4.731445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-0.817871</td>
          <td> 0.500000</td>
        </tr>
        <tr>
          <th>8  </th>
          <td> 1.606445</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-1.130371</td>
          <td> 0.500000</td>
        </tr>
        <tr>
          <th>21 </th>
          <td> 3.798112</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>    rbf</td>
          <td>-1.284538</td>
          <td> 0.500000</td>
        </tr>
      </tbody>
    </table>
    <p>150 rows × 6 columns</p>
    </div>


