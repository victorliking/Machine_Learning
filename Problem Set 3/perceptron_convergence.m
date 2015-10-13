% This function implements the perceptron algorithm.  It is run over different training sets and it
% returns the total number of mistakes reached till the point when a
% continuous set of 'R' examples are found such that there are no mistakes
% on those examples.
%
% Input:
% x: k-by-n matrix,
% y: k-by-1 vector, each element can be 1 or -1

function [wrong] = perceptron_convergence(x,y)
    [m,n] = size(x);
    w = zeros(1,n);   
    theta = 0;
    wrong = 0;
    R = 1000;
    correct = 0;
   
    for iterate = 1:10
        for i = 1:m
            if(correct == R)
                break;
            end
            if((dot(w,x(i,:))+theta) * y(i) <= 0)
                wrong = wrong + 1;
                correct = 0;
                for j = 1:n
                    w(j) = w(j) + 1*y(i)*x(i,j);
                end
                theta = theta + 1*y(i);
            else
                correct = correct + 1;
            end
        end
        if (correct == R)
            break;
        end
    end
    if (correct ~= R)
        disp('reduce R fo perceptron')
    end