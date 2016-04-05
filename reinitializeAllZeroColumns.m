function L  = reinitializeAllZeroColumns( L )
%reinitializeAllZeroColumns Summary of this function goes here
%   Detailed explanation goes here

[rownum , colnum] = size(L);
for j = 1:colnum
    if L(:,j)== 0
       for i=1:rownum
           L(i,j) = rand();
       end
    end
end

