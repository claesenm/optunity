function obj = CrossValidated(fun, x, y, strata, clusters, num_folds, ...
                             num_iter, folds, regenerate_folds, aggregator)


    obj.fun = fun;
    if isempty(x)
        error('x cannot be empty.');
    end
    obj.x = x;
    if isempty(y)
        obj.y = [];
    elseif size(y, 1) == size(x, 1)
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

    obj = class(obj, "CrossValidated");

end
