function folds = get_folds(obj)
    %Returns the folds to be used in cross-validation.
    % The result is a matrix of size num_instances * num_iter.
    if obj.regenerate_folds || isempty(obj.current_folds)
        obj.current_folds = generate_folds(size(obj.x, 1), 'num_folds', obj.num_folds, ...
            'num_iter', obj.num_iter, 'strata', obj.strata, 'clusters', obj.clusters);
    end
    folds = obj.current_folds;
end % get_folds
