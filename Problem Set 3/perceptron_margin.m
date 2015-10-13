% This function implements the perceptron with margin algorithm on the
% training set.
%
% Input:
% x: k-by-n matrix,
% y: k-by-1 vector, each element can be 1 or -1
% r: learning rate.


function [w,theta] = perceptron_margin(x,y,r)
%PERCEPTRON_MARGIN Summary of this function goes here
%   Detailed explanation goes here
   [m,n] = size(x);
    w = zeros(1,n);
    theta = 0;
    for j = 1:20
       for i = 1:m
         if (dot(w,x(i,:))+theta) * y(i) <= 1 
            w = w + r*y(i)*x(i,:);
            theta = theta + r*y(i);
         end
       end
    end

end

