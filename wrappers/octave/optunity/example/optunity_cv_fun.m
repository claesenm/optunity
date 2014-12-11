function [ result ] = optunity_cv_fun( x_train, x_test, pars )

disp('training set:');
disp(x_train');
disp('test set:');
disp(x_test');

result = -pars.x^2 - pars.y^2;

end

