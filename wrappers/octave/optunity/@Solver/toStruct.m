function [struc] = toStruct(obj)
    struc = obj.cfg;
    struc.solver_name = obj.solver_name;
end
