function [ wrong ] = winnow_convergence( x,y,alpha )
%WINNOW_CONVERGENCE Summary of this function goes here
%   Detailed explanation goes here
    
    [m,n] = size(x);
    w = ones(1,n);
    theta = -n;
    wrong =0;
    correct = 0;
    R = 1000;
    
    for iter = 1:10
        for i = 1:m
          if(correct ==R) 
              break;
          end 

          if (dot(w,x(i,:))+theta) * y(i) <= 0
            wrong = wrong+1;
            correct = 0;
            for k = 1:n
                w(k) = w(k) * alpha^(y(i)*x(i,k));
            end
          else 
            correct = correct +1;
          end
        end
        if(correct ==R)
            break;
        end;
    end
    if (correct ~= R)
        disp('reduce the R for the winnow')
    end
end

