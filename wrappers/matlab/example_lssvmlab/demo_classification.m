close all; clear;

%% generate data: a simple double helix
r = 0.5;
h = 1;
helix = @(x, theta, n) [x + r*cos(theta + linspace(0, 4*pi, n)); ...
        r*sin(theta + linspace(0, 4*pi, n)); linspace(0, h, n)]';

noise = 0.1;
ntr = 200;
pos = helix(0, 0, ntr);
neg = helix(0, pi, ntr);
X = [pos; neg] + noise*randn(2*ntr, 3);
Y = [ones(ntr, 1); -1*ones(ntr, 1)];

test_noise = 0.05;
nte = 1000;
pos = helix(0, 0, nte);
neg = helix(0, pi, nte);
Xt = [pos; neg] + test_noise*randn(2*nte, 3);
Yt = [ones(nte, 1); -1*ones(nte, 1)];

type = 'classification';

%% tune with LS-SVMlab
[lssvm_gam, lssvm_sig2] = tunelssvm({X,Y,'c',[],[],'RBF_kernel'}, 'simplex',...
    'leaveoneoutlssvm', {'misclass'});
[alpha_lssvm,b_lssvm] = trainlssvm({X,Y,type,lssvm_gam,lssvm_sig2,'RBF_kernel'});
disp(['LS-SVMlab tuning results: gam=',num2str(lssvm_gam),', sig2=',num2str(lssvm_sig2)]);


%% tune with Optunity
% objective function: 10-fold cross-validated mse
obj_fun = optunity.cross_validate(@demo_classification_misclass, X, 'y', Y, 'num_folds', 10);
% perform tuning, using 100 function evaluations:
%   1 < gam < 100
%   0.01 < sig2 < 2
opt_pars = optunity.minimize(obj_fun, 100, 'gam', [1, 100], 'sig2', [0.01, 2]);
[alpha_optunity,b_optunity] = trainlssvm({X,Y,type,opt_pars.gam,opt_pars.sig2,'RBF_kernel'});
disp(['Optunity tuning results: gam=',num2str(opt_pars.gam),', sig2=',num2str(opt_pars.sig2)]);

%% predict with both models
Yt_lssvm = simlssvm({X,Y,type,lssvm_gam,lssvm_sig2,'RBF_kernel','preprocess'},{alpha_lssvm,b_lssvm},Xt);
Yt_optunity = simlssvm({X,Y,type,opt_pars.gam,opt_pars.sig2,'RBF_kernel','preprocess'},{alpha_optunity,b_optunity},Xt);
 
disp(['Accuracy LS-SVM: ', num2str(sum(Yt==Yt_lssvm)/numel(Yt))]);
disp(['Accuracy Optunity: ', num2str(sum(Yt==Yt_optunity)/numel(Yt))]);

%% plot test predictions after tuning with LS-SVMlab
figure; hold on;
plot3(Xt(Yt_lssvm==1,1), Xt(Yt_lssvm==1,2), Xt(Yt_lssvm==1,3), '.b');
plot3(Xt(Yt_lssvm==-1,1), Xt(Yt_lssvm==-1,2), Xt(Yt_lssvm==-1,3), '.r');
title('LS-SVMlab-tuned test predictions');

%% plot test predictions after tuning with Optunity
figure; hold on;
plot3(Xt(Yt_optunity==1,1), Xt(Yt_optunity==1,2), Xt(Yt_optunity==1,3), '.b');
plot3(Xt(Yt_optunity==-1,1), Xt(Yt_optunity==-1,2), Xt(Yt_optunity==-1,3), '.r');
title('Optunity-tuned test predictions');