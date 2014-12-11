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
