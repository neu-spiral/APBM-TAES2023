% Implements the APBM proposed in
% Imbiriba et. al., Hybrid Neural Network Augmented Physics-based Models
% for Nonlinear Filtering. Using PVT data.
%
% Author: Tales Imbiriba.

clear all; close all;

use_more_accurate_data  = false;

if use_more_accurate_data
    load PVT_more_accurate.mat
    q = sqrt(1e-1);    % for more accurate data
    r = sqrt(2);         % noise covariance      % for more accurate data
else
    load PVT_less_accurate.mat
    q = sqrt(1e-3);  % for less accurate data
    r = sqrt(10);         % noise covariance   % for less accurate data
end

% centering the measurements around zero. 
m_x = mean(pos_x);
m_y = mean(pos_y);
pos_x = pos_x - m_x;
pos_y = pos_y - m_y;

Ts=median(RX_time(2:end)- RX_time(1:end-1));
% q = sqrt(1e-2);  % for less accurate data

Q = q^2 * eye(2);

Gamma = [Ts^2/2 0;
        Ts   0;
        0  Ts^2/2;
        0  Ts];
    
% r = sqrt(10);         % noise covariance   % for less accurate data
% r = sqrt(1);         % noise covariance      % for more accurate data
R = r^2 * eye(2);       % noise covariance matrix

x = [pos_x(1) 0 pos_y(1) 0]; P = 10*eye(4);             % initializing variables used in the data 
x_dim = 4;
y_dim = 2;

%% CVT

% defining transition and measurement functions
cvtfunc = @const_vel_transition_function;       % const vel trans. function

% data gen measurement function
hfun = @(x) [x(1), x(3)];

ckf = trackingCKF(cvtfunc, hfun, x, 'ProcessNoise', Gamma*Q*Gamma', 'MeasurementNoise', R, 'StateCovariance', P);

%% APBM 

apbm_hfun = @apbm_reg_measurement_function;
apbm_tfunc = @apbm_transition_function;             % APBM transition function

% APBM initialization 
apbm_nn_mlp = tmlp(length(x), length(x), [2]);         % creating NN object
theta = apbm_nn_mlp.get_params();                        % getting NN parameters
w0 = [1;0] + 1e-2*randn(2,1);
x_nn = [theta; w0; x'];                                  % initial NN_CKF states

% NN process noise
% Q_nn = q^2*eye(length(x_nn));                       
Q_nn = 1e-2*eye(length(x_nn));
Q_nn(end-x_dim+1:end, end-x_dim+1:end) = Gamma*Q*Gamma';

% Initial NN state cov 
P_apbm = 1e-3*eye(length(x_nn));
P_apbm(end-x_dim+1:end, end-x_dim+1:end) = P;

% noise covariance matrix for augmented likelihood model (for
% regularization)
lambda = 1e8;
R_apbm = (1/lambda)*eye(length(x_nn)-2);
R_apbm(end-y_dim+1:end,end-y_dim+1:end) = R;

% create CKF filter
apbm_ckf = trackingCKF(apbm_tfunc, apbm_hfun, x_nn, 'ProcessNoise', Q_nn, 'MeasurementNoise', R_apbm, 'StateCovariance', P_apbm);



%% Loop

N = length(pos_x);
% N = 1000;
save_pos_cv = zeros(N, 2);
save_pos_apbm = zeros(N, 2);

% zero vector for likelihood augmentation
zero_meas = zeros(apbm_nn_mlp.nparams,1);

for n=1:N
    if mod(n,1000)==0
        sprintf("n = %d\r", n)
    end
   % measurement
    y = [pos_x(n), pos_y(n)]';
   
   % standard CKF (constant velocity)
   [ckf_xPred, ckf_pPred] = predict(ckf, Ts);
   [ckf_xCorr, ckf_pCorr] = correct(ckf, y);
   save_pos_cv(n,:) = [ckf_xCorr(1), ckf_xCorr(3)];
   
%    % APBM 
   [apbm_xPred, apbm_pPred] = predict(apbm_ckf, Ts, apbm_nn_mlp);
   % correct with augmented likelihood function:
   [apbm_ckf_xCorr, apbm_ckf_pCorr] = correct(apbm_ckf, [zero_meas; 1; 0; y], apbm_nn_mlp);     
    
   x_apbm = apbm_ckf_xCorr(end-x_dim+1:end);
   save_pos_apbm(n,:) = [x_apbm(1), x_apbm(3)];
end

%% Plotting
fontsize=16;
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

figure; 
plot(pos_x(1:N), pos_y(1:N), '.', 'color',[.8,.8,.8])
hold on;
plot(save_pos_cv(:,1), save_pos_cv(:,2), '-k', 'linewidth', 1.5)
plot(save_pos_apbm(:,1), save_pos_apbm(:,2), '-b', 'linewidth', 1.5)
grid
ax = gca; ax.FontSize = fontsize-2;
legend('measurements','CV', 'APBM', 'fontsize', fontsize-4)

%% figure
fontsize = 28;
h = figure;
[latitudeECEF_deg, longitudeECEF_deg, altitudeECEF_m] = ecef2geo([pos_x'+m_x,pos_y'+m_y,pos_z'], 1);
geoplot(latitudeECEF_deg(1:3:end),longitudeECEF_deg(1:3:end),'.', 'color',[.9,.4,.5])
hold on 

[latitudeECEF_deg, longitudeECEF_deg, altitudeECEF_m] = ecef2geo([save_pos_cv(:,1)+m_x,save_pos_cv(:,2)+m_y,pos_z'], 1);
geoplot(latitudeECEF_deg,longitudeECEF_deg,'k-','LineWidth',1)

[latitudeECEF_deg, longitudeECEF_deg, altitudeECEF_m] = ecef2geo([save_pos_apbm(:,1)+m_x,save_pos_apbm(:,2)+m_y,pos_z'], 1);
geoplot(latitudeECEF_deg,longitudeECEF_deg,'b-','LineWidth',1)
% geobasemap streets
% geobasemap streets-light
geobasemap topographic
ax = gca; ax.FontSize = fontsize-2;
% xlabel('longitude')
% ylabel('latitude')
legend('measurements','CV', 'APBM', 'fontsize', fontsize-4)
% exportgraphics(ax, 'figs/geomap_apbm_cv2.pdf')

%% Auxiliary Functions

function [x] = const_vel_transition_function(x_prev, Ts)
    F = [1 Ts 0 0;
         0 1 0 0;
         0 0 1 Ts;
         0 0 0 1];
     x = F*x_prev;
end

function [x] = apbm_transition_function(x_prev, Ts, nn_mlp)
    % x_prev = [theta_prev, w_prev; s_prev]
%     global nn_mlp
    F = [1 Ts 0 0;
     0 1 0 0;
     0 0 1 Ts;
     0 0 0 1];
    theta = x_prev(1:nn_mlp.nparams);
    w = x_prev(nn_mlp.nparams + 1: nn_mlp.nparams + 2);
    s = x_prev(nn_mlp.nparams + 3: end);
    nn_mlp.set_params(theta)
    s = w(1)*F*s + w(2)*nn_mlp.forward(s);
    x = [theta; w; s];
end

function y = apbm_reg_measurement_function(x, nn_mlp)
%     global nn_mlp
    theta = x(1:nn_mlp.nparams);
    w = x(nn_mlp.nparams + 1: nn_mlp.nparams + 2);
    s = x(nn_mlp.nparams + 3: end);
%     y = [30 - 10*log10(norm(-s(1:2:3))^2.2); atan2(s(3),s(1))];
    y = [s(1), s(3)]';
    y = [theta; w; y];
end


