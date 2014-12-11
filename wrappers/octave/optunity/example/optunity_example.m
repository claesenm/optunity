close all; clear all;
drawfig = true;

%% print general manual and obtain list of solvers
solvers = manual();

%% target function: f(x,y) = - x^2 - y^2
offx = rand();
offy = rand();
f = @(pars) - (offx+pars.x)^2 - (offy+pars.y)^2;

%% cross-validation example
%global DEBUG_OPTUNITY;
%DEBUG_OPTUNITY = true;
strata = {[1,2,3], [6,7,8,9]};
folds = generate_folds(20, 'num_folds', 10, 'num_iter', 2, 'strata', strata)

%% optimize using grid-search
grid_solver = make_solver('grid search','x', -5:0.5:5, 'y', -5:0.5:5);
[grid_solution, grid_details] = optimize(grid_solver, f);

%% simple API
% maximization
[max_solution, max_details, max_solver] = maximize(f, 200, 'solver_name', 'random search', 'x', [-5, 5], 'y', [-5, 5]);
% minimization
[min_solution, min_details, min_solver] = minimize(f, 200, 'x', [-5, 5], 'y', [-5, 5]);

%% optimize using random-search
rnd_solver = make_solver('random search', 'x', [-5, 5], 'y', [-5, 5], 'num_evals', 400);
[rnd_solution, rnd_details] = optimize(rnd_solver, f);

%% check if the nelder-mead solver is available in the list of solvers
nm_available = any(arrayfun(@(x) strcmp(x, 'nelder-mead'), solvers));

%% optimize using nelder-mead if it is available
nm_solver = make_solver('nelder-mead', 'x', 4,'y', -4, 'ftol', 1e-7);
[nm_solution, nm_details] = optimize(nm_solver, f);

%% check if PSO is available
pso_solver = make_solver('particle swarm', 'num_particles', 5, 'num_generations', 30, ...
    'x', [-5, 5], 'y', [-5, 5], 'max_speed', 0.03);
[pso_solution, pso_details] = optimize(pso_solver, f);

%% check if CMA-ES is available
cma_available = any(arrayfun(@(x) strcmp(x, 'cma-es'), solvers));
if cma_available
    cma_solver = make_solver('cma-es', 'num_generations', 25, ...
        'sigma', 5, 'x', 2, 'y', 4);
    [cma_solution, cma_details] = optimize(cma_solver, f);
end

%% draw a figure to illustrate the call log of all solvers
if drawfig
    figure; hold on;
    plot(grid_details.call_log.args.x, grid_details.call_log.args.y, 'r+','LineWidth', 2);
    plot(rnd_details.call_log.args.x, rnd_details.call_log.args.y, 'k+','LineWidth', 2);
    plot(nm_details.call_log.args.x, nm_details.call_log.args.y, 'm', 'LineWidth', 3);
    plot(pso_details.call_log.args.x, pso_details.call_log.args.y, 'bo', 'LineWidth', 2);
    if cma_available
        plot(cma_details.call_log.args.x, cma_details.call_log.args.y, 'go', 'LineWidth', 2);
    end    
    [X,Y] = meshgrid(-5:0.1:5);
    Z = arrayfun(@(idx) f(struct('x',X(idx),'y',Y(idx))), 1:numel(X));
    Z = reshape(Z, size(X,1), size(X,1));
    contour(X,Y,Z);
    axis square;
    xlabel('x');
    ylabel('y');
    title('f(x,y) = -x^2-y^2');
    axis([-5.5, 5.5, -5.5, 5.5])
    legends = {['grid search (',num2str(grid_details.stats.num_evals),' evals)'], ...
             ['random search (',num2str(rnd_details.stats.num_evals),' evals)'], ...
        };
    
    legends{end+1} = ['Nelder-Mead (',num2str(nm_details.stats.num_evals),' evals)'];
    legends{end+1} = ['particle swarm (',num2str(pso_details.stats.num_evals),' evals)'];
    if cma_available
        legends{end+1} = ['CMA-ES (',num2str(cma_details.stats.num_evals),' evals)'];
    end
    legend(legends, -1);
    
    num_evals = [grid_details.stats.num_evals, rnd_details.stats.num_evals];
    optima = [grid_details.optimum, rnd_details.optimum];
    ticks = {'grid search', 'random search'};
    num_evals(end+1) = nm_details.stats.num_evals;
    optima(end+1) = nm_details.optimum;
    ticks{end+1} = 'Nelder-Mead';
    num_evals(end+1) = pso_details.stats.num_evals;
    optima(end+1) = pso_details.optimum;
    ticks{end+1} = 'particle swarm';
    if cma_available
       num_evals(end+1) = cma_details.stats.num_evals;
       optima(end+1) = cma_details.optimum;
       ticks{end+1} = 'CMA-ES';
    end
    
    figure; hold on;
    
    optima(optima~=0) = log10(abs(optima(optima~=0)));
    bar(optima);
    set(gca,'XTick', 1:numel(optima), 'XTickLabel',ticks);
    xlabel('solver');
    ylabel('log10(error) or 0 if exact');
    
end

%% grid-search with constraints and defaulted function value -> see call log 
s_oo1 = make_solver('grid search', 'x', -5:0.5:5, 'y', -5:0.5:5);
constraints = struct('ub_o', struct('x', 3));
[constr_solution, constr_details] = optimize(s_oo1, f, 'constraints', constraints, 'default', -100);

%% grid-search with warm start: already evaluated grid -> warm_nevals = 0
s_oo2 = make_solver('grid search', 'x', [1, 2], 'y', [1, 2]);
call_log = struct('args',struct('x',[1 1 2 2], 'y', [1 2 1 2]), ...
    'values',[1 2 3 4]);
[warm_solution, warm_details] = optimize(s_oo2, f, 'call_log', call_log);



%% cross-validation
x = (1:10)';
cvf = cross_validate(@optunity_cv_fun, x);
performance = cvf(struct('x',1,'y',2));
