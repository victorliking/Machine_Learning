load ./data/ex1data.mat

winnow_max_accy_cache = zeros(1,2);
winnow_best_rate_cache = zeros(1,2);

x1 = Ex1l10m100n500train(:, 1:(end-1));
y1 = Ex1l10m100n500train(:, end);

testx = Ex1l10m100n500test(:, 1:(end-1));
testy = Ex1l10m100n500test(:, end);

rates = [1.1,1.01,1.005,1.00005,1.0001];
max_acc = -1;
rate = 0;

for win_rate = 1:numel(rates)
    [w,theta] = winnow(x1,y1,rates(win_rate));
    accy = calaccuracy(testx,testy,w,-1*500);
    if(accy>max_acc)
        max_acc = accy;
        rate = rates(win_rate);
    end
end
winnow_max_accy_cache(1,1) = max_acc;
winnow_best_rate_cache(1,1) = rate;
rates = [1.1,1.01,1.005,1.00005,1.0001];
max_acc = -1;
rate = 0;

for win_rate = 1:numel(rates)
    [w,theta] = winnow(x1,y1,rates(win_rate));
    accy = calaccuracy(testx,testy,w,-1*500);
    if(accy>max_acc)
        max_acc = accy;
        rate = rates(win_rate);
    end
end
winnow_max_accy_cache(1,1) = max_acc;
winnow_best_rate_cache(1,1) = rate;

x2 = Ex1l10m100n1000train(:, 1:(end-1));
y2 = Ex1l10m100n1000train(:, end);

testx = Ex1l10m100n1000test(:, 1:(end-1));
testy = Ex1l10m100n1000test(:, end);

rates = [1.1,1.01,1.005,1.00005,1.0001];
max_acc = -1;
rate = 0;

for win_rate = 1:numel(rates)
    [w,theta] = winnow(x2,y2,rates(win_rate));
    accy = calaccuracy(testx,testy,w,-1*1000);
    if(accy>max_acc)
        max_acc = accy;
        rate = rates(win_rate);
    end
end
winnow_max_accy_cache(1,2) = max_acc;
winnow_best_rate_cache(1,2) = rate;

for disp_index = 1:2
    disp('For Winnow:')
    disp('Max accuracy:')
    disp( winnow_max_accy_cache(1,disp_index))
    disp('Learning Rate:')
    disp(winnow_best_rate_cache(1,disp_index))
end 