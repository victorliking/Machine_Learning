function [ mistakes ] = perceptron_mistake(x,y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [m,n] = size(x);
    mistakes = zeros(1,(m/100));
    ret =0;
    w = zeros(1,n);
    theta = 0;
    for i = 1:m
        if(mod(i,100)==0)
            mistakes(1,i/100) =ret;
        end
        if (dot(w,x(i,:))+theta) * y(i) <= 0
            ret = ret+1;
            w = w + 1*y(i)*x(i,:);
            theta = theta + 1*y(i);
        end
    end
end

