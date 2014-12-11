function result = feval(obj, pars)
    %Performs cross-validation as configured with hyperparameters pars and returns the
    %result.
    if ~isa(obj.fun, 'function_handle')
        error('Internal function to be cross-validated is not set.');
    end
    folds = get_folds(obj);
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
