function [w_err, weights] = batch_learn(HW1_sortdat, weights,ALPHA)
%BATCH_LEARN Summary of this function goes here
%   Detailed explanation goes here

%% CONSTANTS
THETA = 0;

%% Variables
x_i = HW1_sortdat(:,1:end-1);
t = HW1_sortdat(:,end);
y = sum(x_i .* weights.',2) >= THETA;

%% new_weight = w_i + alpha*(t-y)*x_i

err = t - y;
w_err = sum(abs(err));

dw = ALPHA * err .* x_i;

dw_avg = mean(dw,1);

weights = weights + dw_avg.';

end