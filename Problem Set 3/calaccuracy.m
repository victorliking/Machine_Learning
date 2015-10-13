% This function is used to find the accuracy of our algorithm over the 
% testing data.

function [accuracy] = calaccuracy(x,y,w,theta)
%CALACCURACY Summary of this function goes here
%   Detailed explanation goes here
    [m,n] = size(x);
    correct = 0;
    for i = 1:m
        if y(i)*(dot(w,x(i,:)) + theta) > 0
        correct = correct+1;
        end
    end
    accuracy = 100*(correct/m);
end

