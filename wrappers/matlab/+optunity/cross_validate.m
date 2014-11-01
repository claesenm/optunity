function cv = cross_validate(fun, x, varargin)
%CROSS_VALIDATE: Decorates a function to perform cross-validation when
%evaluated.
%
%This function accepts the following arguments:
%- fun: the function that should be evaluated through cross-validation
%      This function must accept one of the following argument lists:
%          fun(x_train, x_test, pars)
%          fun(x_train, y_train, x_test, y_test, pars)
%      The first form for unsupervised algorithms, second for supervised.
%      x_* are data, y_* are labels and pars is a struct of
%      hyperparameters.
%- x: the data set (excluding labels)
%- varargin: a list of optional key:value pairs to further configure
%      the cross-validation procedure
%  - y: labels, if specified this must have same number of rows as x
%      default: empty
%  - num_folds: number of folds to use in cross-validation
%      default: 10
%  - num_iter: number of cross-validation iterations to perform 
%      default: 1
%  - strata: cell array containing strata, e.g. indices of instances that
%      must be spread out across folds (default: empty)
%  - clusters: cell array containing clusters, e.g. indices of instances
%      that must be kept within a single fold
%      default: empty
%  - folds: matrix of size num_instances * num_iter containing
%      prespecified folds to use in cross-validation
%      default: empty
%  - regenerate_folds: boolean, whether or not folds must be regenerated
%      at every cross-validation
%      default: false
%  - aggregator: function handle containing the function to be used to
%      aggregate results across folds
%      default: mean
%
%This returns an optunity.CrossValidated object, which is a function that
%accepts a struct defining the hyperparameters. Every evaluation will
%perform cross-validation as configured.

%% process varargin
defaults = struct('num_folds', 10, 'y', [], 'strata', [], ...
    'folds', [], 'num_iter', 1, 'regenerate_folds', false, ...
    'clusters', [], 'aggregator', @mean);
options = optunity.process_varargin(defaults, varargin, false);

cv = optunity.CrossValidated(fun, x, options.y, options.strata, options.clusters, ...
    options.num_folds, options.num_iter, options.folds, options.regenerate_folds, ...
    options.aggregator);

end