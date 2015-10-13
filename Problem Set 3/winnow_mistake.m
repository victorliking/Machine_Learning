function [ mistakes ] = winnow_mistake(x,y,alpha )
%WINNOW_MISTAKE Summary of this function goes here
%   Detailed explanation goes here
    [m,n] = size(x);
    w = ones(1,n);
    theta = -n;
    mistakes = zeros(1,(m/100));
    ret =0;
    for i = 1:m
      if(mod(i,100)==0) 
          mistakes(1,i/100) = ret;
      end 
      
      if (dot(w,x(i,:))+theta) * y(i) <= 0
        ret = ret+1;
        for k = 1:n
            w(k) = w(k) * alpha^(y(i)*x(i,k));
        end
      end
    end

end

