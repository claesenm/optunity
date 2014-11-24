function [ misclass ] = demo_classification_misclass( x_train, y_train, x_test, y_test, pars )
%demo_classification_misclass Example objective function to use with Optunity's
%cross-validation. We tune an LS-SVM with RBF kernel for classification.
%   Arguments:
%   - x_train: training data
%   - y_train: training labels
%   - x_test: test data
%   - y_test: test labels
%   - pars: struct of hyperparameters (gam and sig2)

% train model
[alpha,b] = trainlssvm({x_train,y_train,'classification',pars.gam,pars.sig2,'RBF_kernel'});

% predict test data
Yt = simlssvm({x_train,y_train,'classification',pars.gam,pars.sig2,'RBF_kernel','preprocess'},{alpha,b},x_test);

% compute misclassification rate
misclass = sum(Yt ~= y_test)/numel(Yt);
end

