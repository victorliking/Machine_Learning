function [ wrong ] = adagrad_convergence( x,y,r )
%ADAGRAD_CONVERGENCE Summary of this function goes here
%   Detailed explanation goes here
        [m,n] = size(x);
        w = zeros(1,n);   
        theta = 0;
        G = zeros(1,n+1);
        wrong = 0;
        g = zeros(m,n+1);
        correct =0;
        R = 1000;
      for iter = 1:10  
        for i = 1:m
            if(correct==R) %we get the convergence here 
                break;
            end
            if(dot(w(1:n),x(i,:))+theta) * y(i) <=0
                wrong = wrong+1;
                correct =0;
            else 
                correct = correct +1;
            end
            if(dot(w(1:n),x(i,:))+theta) * y(i) <= 1
                for j = 1:n
                    g(i,j) = -1*y(i)*x(i,j);
                end
                g(i,n+1) = -1*y(i);
            end   
            for j = 1:n+1
                G(j) = G(j) + g(i,j).^2;
            end

            if(dot(w(1:n),x(i,:))+theta) * y(i) <=1    
                for j = 1:n
                    if (G ~= 0)
                        w(j) = w(j) + r*y(i)*x(i,j)/(sqrt(G(j)));
                    end
                end
                theta = theta + r*y(i)*1/sqrt(G(n+1));
            end
        end
     if(correct==R)
         break;
     end
      end
    if (correct ~= R)
        disp('reduce R for adagrad convergence')
    end

