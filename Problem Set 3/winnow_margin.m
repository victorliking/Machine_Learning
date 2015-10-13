% This function implements the winnow with margin algorithm for the
% training data.

function [w] = winnow_margin(x,y,alpha,margin)
%WINNOW_MARGIN Summary of this function goes here
%   Detailed explanation goes here
    [m,n] = size(x);
    w = ones(1,n);
    theta = -1*n;
    for j = 1:20
       for i = 1:m
         if (dot(w,x(i,:))+theta) * y(i) <= margin
            for k = 1:n
                w(k) = w(k) * alpha^(y(i)*x(i,k));
            end
         end
       end
    end

end

