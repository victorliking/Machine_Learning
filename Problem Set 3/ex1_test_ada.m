load ./data/ex1data.mat


adagrad_max_accy_cache = zeros(1,2);
adagrad_best_rate_cache = zeros(1,2);

x1 = Ex1l10m100n500train(:, 1:(end-1));
y1 = Ex1l10m100n500train(:, end);

testx = Ex1l10m100n500test(:, 1:(end-1));
testy = Ex1l10m100n500test(:, end);


rates = [1.5,0.25,0.03,0.05,0.001];
max_acc = -1;
rate = 0;

for ada_rate = 1:numel(rates)
    [w,theta] = adagrad(x1,y1,rates(ada_rate));
    accy = calaccuracy(testx,testy,w,theta);
    if(accy>max_acc)
        max_acc = accy;
        rate = rates(ada_rate);
    end
end
adagrad_max_accy_cache(1,1) = max_acc;
adagrad_best_rate_cache(1,1) = rate;


x2 = Ex1l10m100n1000train(:, 1:(end-1));
y2 = Ex1l10m100n1000train(:, end);

testx = Ex1l10m100n1000test(:, 1:(end-1));
testy = Ex1l10m100n1000test(:, end);

rates = [1.5,0.25,0.03,0.05,0.001];
max_acc = -1;
rate = 0;

for ada_rate = 1:numel(rates)
    [w,theta] = adagrad(x2,y2,rates(ada_rate));
    accy = calaccuracy(testx,testy,w,theta);
    if(accy>max_acc)
        max_acc = accy;
        rate = rates(ada_rate);
    end
end
adagrad_max_accy_cache(1,2) = max_acc;
adagrad_best_rate_cache(1,2) = rate;

for disp_index = 1:2
    disp('For ada:')
    disp('Max accuracy:')
    disp( per_max_accy_cache(1,disp_index))
    disp('Learning Rate:')
    disp(per_best_rate_cache(1,disp_index))
end 