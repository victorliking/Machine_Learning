% This function implements the perceptron algorithm on the training set.

function [w,theta] = perceptron(x,y)
%PERCETRON Summary of this function goes here
%   Detailed explanation goes here
    [m,n] = size(x);
    w = zeros(1,n);
    theta = 0;
    for j = 1:20
       for i = 1:m
         if (dot(w,x(i,:))+theta) * y(i) <= 0
            w = w + 1*y(i)*x(i,:);
            theta = theta + 1*y(i);
         end
       end
    end    

end

