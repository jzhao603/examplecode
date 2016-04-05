function [ theta_t, D_t ] = singleTaskLearner( X_t,y_t )
%singleTaskLearner Summary of this function goes here
%   Detailed explanation goes here

[row,n_t] = size(X_t);
theta_t = inv(X_t * X_t')*X_t*y_t;
D_t = (X_t * X_t')/(2*n_t);

end

