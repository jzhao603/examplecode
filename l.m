function f = l( x,L,theta,D)
%l_LSTD Summary of this function goes here
%   Detailed explanation goes here
%u=1e-1;
u=0.5;

f = u* norm(x,1)+ (theta-L*x)'*D*(theta-L*x);

end

