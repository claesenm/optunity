function [ result ] = cv_fun( x_train, x_test, pars )
%CV_FUN Summary of this function goes here
%   Detailed explanation goes here

disp('training set:');
disp(x_train');
disp('test set:');
disp(x_test');

result = -pars.x^2 - pars.y^2;

end

