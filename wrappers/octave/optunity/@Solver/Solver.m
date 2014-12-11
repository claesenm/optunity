function obj = Solver(name, cfg)
   % SOLVER Handle for an Optunity solver.
   % We recommend constructing Solver objects via make_solver().
   % Please refer to the documentation thereof for further details.
   %
   % Solver objects have a maximize(f, varargin) function, offering
   % equivalent functionality to maximize(solver, f, varargin).
   % Please refer to the documentation of maximize() for details.

    obj = struct('cfg', cfg, 'solver_name', name);
    obj = class(obj, "Solver");
end
