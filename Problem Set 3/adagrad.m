% This function implements the adagrad algorithm on the training data.
function [w,theta] = adagrad(x,y,r)
    [k,n] = size(x);
    w = zeros(1,n);   
    theta = 0;
    g = zeros(k,n+1);
    G = zeros(1,n+1);
    
    for iterate = 1:20
        for i = 1:k

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
    end