function [ wrong ] = perceptron_margin_convergence( x,y,r )
%PERCEPTRON_MARGIN_CONVERGENCE Summary of this function goes here
%   Detailed explanation goes here
 [m,n] = size(x);
    w = zeros(1,n);
    theta = 0;
    wrong =0 ;
    R =1000;
    correct = 0;
    for iter = 1:10
        for i = 1:m
          if(correct==R)
              break;
          end
          if(dot(w,x(i,:))+theta) * y(i) <=0
              wrong = wrong+1;
              correct =0 ;
          else 
              correct = correct +1 ;
          end
          if (dot(w,x(i,:))+theta) * y(i) <= 1 
            w = w + r*y(i)*x(i,:);
            theta = theta + r*y(i);
          end
        end
        if(correct == R)
            break;
        end;
    end
    if(correct~=R)
        disp('reduce the R in perceptron margin')
    end
end
