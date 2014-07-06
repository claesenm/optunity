close all; clear all;
drawfig = true;

%% target function: f(x,y) = - x^2 - y^2
f = @(pars) - pars.x^2 - pars.y^2;

%% cross-validation example
strata = {[1,2,3], [6,7,8,9]};
folds = optunity.cross_validation(20, 'num_folds', 10, 'num_iter', 2, 'strata', strata);

%% optimize using grid-search
solver_config = struct('x', -5:0.5:5, 'y', -5:0.5:5);
[grid_solution, grid_optimum, grid_nevals, grid_log] = ...
    optunity.solve('grid-search', solver_config, f, 'return_call_log', true);

%% optimize using random-search
solver_config = struct('x',[-5, 5], 'y', [-5, 5], 'num_evals', 100);
[rnd_solution, rnd_optimum, rnd_nevals, rnd_log] = ...
    optunity.solve('random-search', solver_config, f, 'return_call_log', true);

%% check if the nelder-mead solver is available in the list of solvers
solvers = optunity.manual(); % obtain a list of available solvers
nm_available = any(arrayfun(@(x) strcmp(x, 'nelder-mead'), solvers));

%% optimize using nelder-mead if it is available
if nm_available
    solver_config = struct('x0', struct('x',5,'y',-5), 'xtol', 1e-5);
    [nm_solution, nm_optimum, nm_nevals, nm_log] = ...
        optunity.solve('nelder-mead', solver_config, f, 'return_call_log', true);
end

%% draw a figure to illustrate the call log of all solvers
if drawfig
    figure; hold on;
    plot(grid_log.args.x, grid_log.args.y, 'r+','LineWidth', 2);
    plot(rnd_log.args.x, rnd_log.args.y, 'k+','LineWidth', 2);
    if nm_available
        plot(nm_log.args.x, nm_log.args.y, 'b', 'LineWidth', 3);
    end
    [X,Y] = meshgrid(-5:0.1:5);
    Z = arrayfun(@(idx) f(struct('x',X(idx),'y',Y(idx))), 1:numel(X));
    Z = reshape(Z, size(X,1), size(X,1));
    contour(X,Y,Z);
    axis square;
    xlabel('x');
    ylabel('y');
    title('f(x,y) = -x^2-y^2');
    if nm_available
        legend(['grid search (',num2str(grid_nevals),' evals)'], ...
             ['random search (',num2str(rnd_nevals),' evals)'], ...
             ['Nelder-Mead (',num2str(nm_nevals),' evals)']);
            
    else
        legend('grid search', 'random search');
    end
end

%% grid-search with constraints and defaulted function value -> see call log 
solver_config = struct('x', -5:0.5:5, 'y', -5:0.5:5);
constraints = struct('ub_o', struct('x', 3));
[constr_solution, constr_optimum, constr_nevals, constr_log] = ...
    optunity.solve('grid-search', solver_config, f, ...
    'return_call_log', true, 'constraints', constraints, 'default', -100);

%% grid-search with warm start: already evaluated grid -> warm_nevals = 0
solver_config = struct('x', [1, 2], 'y', [1, 2]);
call_log = struct('args',struct('x',[1 1 2 2], 'y', [1 2 1 2]), ...
    'values',[1 2 3 4]);
[warm_solution, warm_optimum, warm_nevals, warm_log] = ...    
    optunity.solve('grid-search', solver_config, f, ...
    'return_call_log', true, 'call_log', call_log);