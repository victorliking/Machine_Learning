function [ mistakes ] = perceptron_margin_mistake( x,y,r )
%PERCEPTRON_MARGIN_MISTAKE Summary of this function goes here
%   Detailed explanation goes here
    [m,n] = size(x);
    w = zeros(1,n);
    theta = 0;
    mistakes = zeros(1,(m/100));
    wrong =0 ;
    for i = 1:m
      if(mod(i,100)==0)
          mistakes(1,i/100)=wrong;
      end
      if(dot(w,x(i,:))+theta) * y(i) <=0
          wrong = wrong+1;
      end
      if (dot(w,x(i,:))+theta) * y(i) <= 1 
        w = w + r*y(i)*x(i,:);
        theta = theta + r*y(i);
      end
    end


end

