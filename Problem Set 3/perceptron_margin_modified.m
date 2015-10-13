function [ w,theta ] = perceptron_margin_modified( x,y,r )
%PERCEPTRON_MARGIN_MODIFIED Summary of this function goes here
%   Detailed explanation goes here
    [k,n] = size(x);
    w = zeros(1,n);   
    theta = 0;
    
    for iterate = 1:20
        for i = 1:k
            if(y(i) == 1)
                if((dot(w,x(i,:))+theta) * y(i) <= 1)
                    w = w + r*y(i)*x(i,:);
                    theta = theta + r*y(i);
                end
            else
                if((dot(w,x(i,:))+theta) * y(i) <= 1/9)
                    w = w + r*y(i)*x(i,:);
                    theta = theta + r*y(i);
                end
            end
        end
    end

end

