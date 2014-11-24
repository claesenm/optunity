function [ mse ] = demo_regression_mse( x_train, y_train, x_test, y_test, pars )
%demo_regression_mse Example objective function to use with Optunity's
%cross-validation. We tune an LS-SVM with RBF kernel for regression.
%   Arguments:
%   - x_train: training data
%   - y_train: training labels
%   - x_test: test data
%   - y_test: test labels
%   - pars: struct of hyperparameters (gam and sig2)

% train model
type = 'function estimation';
[alpha,b] = trainlssvm({x_train,y_train,type,pars.gam,pars.sig2,'RBF_kernel'});

% predict test data
Yt = simlssvm({x_train,y_train,type,pars.gam,pars.sig2,'RBF_kernel','preprocess'},{alpha,b},x_test);

% compute mse
mse = sum((y_test - Yt).^2) / numel(y_test);

end

