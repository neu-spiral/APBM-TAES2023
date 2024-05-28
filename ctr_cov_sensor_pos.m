function [R_k] = ctr_cov_sensor_pos(xpred,R , SigmaRadarLoc)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    % measurement eq. linearisation for non-additive measurement noise
    % - Stirling interpolation, differences instead of derivatives

    % state estimate (posx, velx, posy, vely)
    % xpred = [100; 1; 200; 0.5];
%     xpred = [10; 1; 20; 0.5];
    % xpred = [1; 1; 2; 0.5];

    % radar position uncertainty xi (assuming radar position [0, 0])
%     SigmaRadarLoc = [1 0.2; 0.2 1.8];

    % nonlinear function
    hfun = @(x) [30 - 10*log10(norm(-x(1:2:3))^2.2); atan2(x(3),x(1))];
    % hfun = @(x) x(1:2:3);

    % nonlinear function linearisation wrt non-additive noise "xi" (with mean "ximean")
    % z = h(x,xi) + v ~ h(x,xiMean) + J(x,ximean)(xi-ximean) + v

    % *** Taylor expansion/Stirling' interpolation absed integration ***
    % Approximate "Jacobian" calculation (evaluated at xpred and non-additive noise mean)
    % - first order central differences
    % - dh1/dxi1, dh2/dxi1
    deltaxi = [sqrt(SigmaRadarLoc(1,1)); 0];
    auxPosPlus = zeros(4,1);
    auxPosMinus = zeros(4,1);
    auxPosPlus(1) = xpred(1) + deltaxi(1);
    auxPosPlus(3) = xpred(3) + deltaxi(2);
    auxPosMinus(1) = xpred(1) - deltaxi(1);
    auxPosMinus(3) = xpred(3) - deltaxi(2);
    auxhPlus = hfun(auxPosPlus); 
    auxhMinus = hfun(auxPosMinus); 
    J(1,1) = (auxhPlus(1)-auxhMinus(1))/(2*deltaxi(1));
    J(2,1) = (auxhPlus(2)-auxhMinus(2))/(2*deltaxi(1));
    % - dh1/dxi2, dh2/dxi2
    deltaxi = [0; sqrt(SigmaRadarLoc(2,2))];
    auxPosPlus = zeros(4,1);
    auxPosMinus = zeros(4,1);
    auxPosPlus(1) = xpred(1) + deltaxi(1);
    auxPosPlus(3) = xpred(3) + deltaxi(2);
    auxPosMinus(1) = xpred(1) - deltaxi(1);
    auxPosMinus(3) = xpred(3) - deltaxi(2);
    auxhPlus = hfun(auxPosPlus); 
    auxhMinus = hfun(auxPosMinus); 
    J(1,2) = (auxhPlus(1)-auxhMinus(1))/(2*deltaxi(2));
    J(2,2) = (auxhPlus(2)-auxhMinus(2))/(2*deltaxi(2));

    % moments calculation
    addMeasNoiseCovSI = J*SigmaRadarLoc*J';

    R_k = R + addMeasNoiseCovSI;

end

