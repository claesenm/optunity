classdef CrossValidated < handle
    %CROSSVALIDATED : functor to evaluates hyperparameter tuples through
    %cross-validation. Constructing these objects should be done via
    %optunity.cross_validate().
    %
    %This object works as a functor, e.g. you can use typical MATLAB
    %function syntax to perform cross-validation:
    %   obj = optunity.cross_validate(...);
    %   score = obj(hyperparameter_struct);
    %
    %To change the function to be cross-validated, you may use the
    %set_fun() member function. This returns a new CrossValidated object
    %with identical configuration for the new function.
    properties (SetAccess = immutable)
        x = [];
        y = [];
        strata = NaN;
        clusters = NaN;
        num_folds = 0;
        num_iter = 0;
        regenerate_folds = false;
        aggregator = NaN;
        fun = NaN;
    end % immutable properties
    properties (SetAccess = protected)
        current_folds = NaN;
    end % protected properties
    methods
        function obj = CrossValidated(fun, x, y, strata, clusters, num_folds, ...
                num_iter, folds, regenerate_folds, aggregator)
            obj.fun = fun;
            if isempty(x)
                error('x cannot be empty.');
            end
            obj.x = x;
            if isempty(y) || (size(y, 1) == size(x, 1))
                obj.y = y;
            else
                error('y must be empty or have the same number of rows as x.');
            end
            obj.strata = strata;
            obj.clusters = clusters;
            obj.num_folds = num_folds;
            obj.num_iter = num_iter;
            if size(folds, 1) == size(x, 1) || isempty(folds)
                obj.current_folds = folds;
            else
                error('Specified folds are of incorrect size');
            end
            obj.regenerate_folds = regenerate_folds;
            if ~isa(aggregator, 'function_handle')
                obj.aggregator = @mean;
            else
                obj.aggregator = aggregator;
            end
        end % CrossValidated constructor
        function folds = get_folds(obj)
            %Returns the folds to be used in cross-validation.
            % The result is a matrix of size num_instances * num_iter.
            if obj.regenerate_folds || isempty(obj.current_folds)
                obj.current_folds = optunity.generate_folds(size(obj.x, 1), 'num_folds', obj.num_folds, ...
                    'num_iter', obj.num_iter, 'strata', obj.strata, 'clusters', obj.clusters);
            end
            folds = obj.current_folds;
        end % get_folds
        function result = feval(obj, pars)
            %Performs cross-validation as configured with hyperparameters pars and returns the
            %result.
            if ~isa(obj.fun, 'function_handle')
                error('Internal function to be cross-validated is not set.');
            end
            folds = obj.get_folds();
            performances = zeros(obj.num_folds, obj.num_iter);
            for iter = 1:obj.num_iter
                for fold = 1:obj.num_folds
                    x_train = obj.x(folds(:, iter) ~= fold, :);
                    x_test = obj.x(folds(:, iter) == fold, :);
                    if isempty(obj.y)
                        performances(fold, iter) = obj.fun(x_train, x_test, pars);
                    else
                        y_train = obj.y(folds(:, iter) ~= fold, :);
                        y_test = obj.y(folds(:, iter) == fold, :);
                        performances(fold, iter) = obj.fun(x_train, y_train, x_test, y_test, pars);
                    end
                end
            end
            results_per_iter = arrayfun(@(x) obj.aggregator(performances(:, x)), 1:obj.num_iter);
            result = mean(results_per_iter);
        end % feval
        function out = subsref(obj, S)
            % Allow cross-validation to be performed through function call
            % syntax. E.g. obj(pars) will work.
            % http://stackoverflow.com/a/18193231/2148672
            switch S(1).type
                case '.'
                    % call builtin subsref, so we dont break the dot notation
                    out = builtin('subsref', obj, S);
                case '()'
                    out = feval(obj, S.subs{:});
                case '{}'
                    error('Not a supported subscripted reference');
            end
        end % subsref
        function cv = set_fun(obj, fun)
            cv = optunity.CrossValidated(fun, obj.x, obj.y, obj.strata, obj.clusters, ...
                obj.num_folds, obj.num_iter, obj.current_folds, obj.regenerate_folds, ...
                obj.aggregator);
        end % set_fun
    end % methods
end % CrossValidated class
