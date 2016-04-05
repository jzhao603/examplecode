function f1 = calculateS_t( L, theta_t, D_t,k)
%calculateS_t Summary of this function goes here
%   Detailed explanation goes here
% min = fminsearch(@l,[0,0,0,0]);
%k = 4;
s_t = zeros(k, 1);
f1 = fminsearch(@(x) l(x,L,theta_t,D_t), s_t);

end

