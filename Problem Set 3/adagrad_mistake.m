function [ mistakes ] = adagrad_mistake( x,y,r )
%ADAGRAD_MISTAKE Summary of this function goes here
%   Detailed explanation goes here
        [m,n] = size(x);
        w = zeros(1,n);   
        theta = 0;
        G = zeros(1,n+1);
        wrong = 0;
        mistakes = zeros(1,(m/100));
        g = zeros(m,n+1);
        
        for i = 1:m
            if(mod(i,100) == 0)
                mistakes(1,i/100) = wrong;
            end
            if(dot(w(1:n),x(i,:))+theta) * y(i) <=0
                wrong = wrong+1;
            end
            if(dot(w(1:n),x(i,:))+theta) * y(i) <= 1
                for j = 1:n
                    g(i,j) = -1*y(i)*x(i,j);
                end
                %Also update g for theta.
                g(i,n+1) = -1*y(i);
            
            end   
            %Calculate the value of sum of gradients' squares.

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
end

