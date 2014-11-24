close all; clear;

fun = @(X) sinc(X) + 0.03 * sin(15*pi * X);

%% construct the data set
X = randn(200, 1);
Y = fun(X)+0.1*randn(length(X),1);

%% tune with LS-SVMlab
[lssvm_gam, lssvm_sig2] = tunelssvm({X,Y,'f',[],[],'RBF_kernel'}, 'simplex',...
    'leaveoneoutlssvm', {'mse'});
type = 'function estimation';
[alpha_lssvm,b_lssvm] = trainlssvm({X,Y,type,lssvm_gam,lssvm_sig2,'RBF_kernel'});
disp(['LS-SVMlab tuning results: gam=',num2str(lssvm_gam),', sig2=',num2str(lssvm_sig2)]);


%% tune with Optunity
% objective function: 10-fold cross-validated mse
obj_fun = optunity.cross_validate(@demo_regression_mse, X, 'y', Y, 'num_folds', 10);
% perform tuning, using 100 function evaluations:
%   1 < gam < 30
%   0.01 < sig2 < 1
opt_pars = optunity.minimize(obj_fun, 100, 'gam', [1, 30], 'sig2', [0.01, 1]);
[alpha_optunity,b_optunity] = trainlssvm({X,Y,type,opt_pars.gam,opt_pars.sig2,'RBF_kernel'});
disp(['Optunity tuning results: gam=',num2str(opt_pars.gam),', sig2=',num2str(opt_pars.sig2)]);

%% generate test data
Xt = (-3:0.01:3)';

% predict test data
Yt = fun(Xt);
Yt_lssvm = simlssvm({X,Y,type,lssvm_gam,lssvm_sig2,'RBF_kernel','preprocess'},{alpha_lssvm,b_lssvm},Xt);
Yt_optunity = simlssvm({X,Y,type,opt_pars.gam,opt_pars.sig2,'RBF_kernel','preprocess'},{alpha_optunity,b_optunity},Xt);

mse_lssvm = sum((Yt-Yt_lssvm).^2) / numel(Yt);
mse_optunity = sum((Yt-Yt_optunity).^2) / numel(Yt);

%% make a nice plot
figure; hold on; 
plot(Xt, Yt,'k');
plot(Xt, Yt_lssvm, 'r');
plot(Xt, Yt_optunity, 'b');
plot(X, Y, 'b.');
axis([-3, 3, -0.5, 1.5]);
legend('true function', ['LS-SVM (test mse=',num2str(mse_lssvm),')'], ...
    ['Optunity (test mse=',num2str(mse_optunity),')']);
xlabel('X');
ylabel('f(x)');