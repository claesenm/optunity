close all; clear all;
drawfig = true;

%% target function: f(x,y) = - x^2 - y^2
offx = rand();
offy = rand();
f = @(pars) - (offx+pars.x)^2 - (offy+pars.y)^2;

%% cross-validation example
strata = {[1,2,3], [6,7,8,9]};
folds = optunity.generate_folds(20, 'num_folds', 10, 'num_iter', 2, 'strata', strata);

%% optimize using grid-search
solver_config = struct('x', -5:0.5:5, 'y', -5:0.5:5);
[grid_solution, grid_details] = ...
    optunity.solve('grid search', solver_config, f, 'return_call_log', true);

%% optimize using random-search
solver_config = struct('x',[-5, 5], 'y', [-5, 5], 'num_evals', 400);
[rnd_solution, rnd_details] = ...
    optunity.solve('random search', solver_config, f, 'return_call_log', true);

%% check if the nelder-mead solver is available in the list of solvers
solvers = optunity.manual(); % obtain a list of available solvers
nm_available = any(arrayfun(@(x) strcmp(x, 'nelder-mead'), solvers));

%% optimize using nelder-mead if it is available
if nm_available
    solver_config = struct('x0', struct('x',4,'y',-4), 'xtol', 1e-4);
    [nm_solution, nm_details] = ...
        optunity.solve('nelder-mead', solver_config, f, 'return_call_log', true);
end

%% check if PSO is available
pso_available = any(arrayfun(@(x) strcmp(x, 'particle swarm'), solvers));
if pso_available
    solver_config = struct('num_particles', 5, 'num_generations', 30, ...
        'x', [-5, 5], 'y', [-5, 5], 'max_speed', 0.03);
    [pso_solution, pso_details] = ...
        optunity.solve('particle swarm', solver_config, f, 'return_call_log', true);
end

%% draw a figure to illustrate the call log of all solvers
if drawfig
    figure; hold on;
    plot(grid_details.call_log.args.x, grid_details.call_log.args.y, 'r+','LineWidth', 2);
    plot(rnd_details.call_log.args.x, rnd_details.call_log.args.y, 'k+','LineWidth', 2);
    if nm_available
        plot(nm_details.call_log.args.x, nm_details.call_log.args.y, 'm', 'LineWidth', 3);
    end
    if pso_available
        plot(pso_details.call_log.args.x, pso_details.call_log.args.y, 'bo', 'LineWidth', 2);
    end    
    [X,Y] = meshgrid(-5:0.1:5);
    Z = arrayfun(@(idx) f(struct('x',X(idx),'y',Y(idx))), 1:numel(X));
    Z = reshape(Z, size(X,1), size(X,1));
    contour(X,Y,Z);
    axis square;
    xlabel('x');
    ylabel('y');
    title('f(x,y) = -x^2-y^2');
    legends = {['grid search (',num2str(grid_details.stats.num_evals),' evals)'], ...
             ['random search (',num2str(rnd_details.stats.num_evals),' evals)'], ...
        };
    
    if nm_available
        legends{end+1} = ['Nelder-Mead (',num2str(nm_details.stats.num_evals),' evals)'];
    end
    if pso_available
        legends{end+1} = ['particle swarm (',num2str(pso_details.stats.num_evals),' evals)'];
    end
    legend(legends);
    
    num_evals = [grid_details.stats.num_evals, rnd_details.stats.num_evals];
    optima = [grid_details.optimum, rnd_details.optimum];
    ticks = {'grid search', 'random search'};
    if nm_available
       num_evals(end+1) = nm_details.stats.num_evals;
       optima(end+1) = nm_details.optimum;
       ticks{end+1} = 'Nelder-Mead';
    end
    if pso_available
       num_evals(end+1) = pso_details.stats.num_evals;
       optima(end+1) = pso_details.optimum;
       ticks{end+1} = 'particle swarm';
    end
    
    figure; hold on;
    
    optima(optima~=0) = log10(abs(optima(optima~=0)));
    bar(optima);
    set(gca,'XTick', 1:numel(optima), 'XTickLabel',ticks);
    xlabel('solver');
    ylabel('log10(error) or 0 if exact');
    
end

%% grid-search with constraints and defaulted function value -> see call log 
solver_config = struct('x', -5:0.5:5, 'y', -5:0.5:5);
constraints = struct('ub_o', struct('x', 3));
[constr_solution, constr_details] = ...
    optunity.solve('grid search', solver_config, f, ...
    'return_call_log', true, 'constraints', constraints, 'default', -100);

%% grid-search with warm start: already evaluated grid -> warm_nevals = 0
solver_config = struct('x', [1, 2], 'y', [1, 2]);
call_log = struct('args',struct('x',[1 1 2 2], 'y', [1 2 1 2]), ...
    'values',[1 2 3 4]);
[warm_solution, warm_details] = ...    
    optunity.solve('grid search', solver_config, f, ...
    'return_call_log', true, 'call_log', call_log);