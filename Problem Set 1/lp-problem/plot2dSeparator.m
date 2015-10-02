% This function plots the linear discriminant.
% YOU NEED TO IMPLEMENT THIS FUNCTION

function plot2dSeparator(w, theta)
    x = linspace(-3,3,200);
    y = (-w(1)*x-theta)/w(2);
    plot(x,y);
end
