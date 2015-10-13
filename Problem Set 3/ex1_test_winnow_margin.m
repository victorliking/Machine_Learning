load ./data/ex1data.mat
winnow_m_max_accy_cache = zeros(1,2);
winnow_m_best_margin_cache = zeros(1,2);
winnow_m_best_alpha_cache = zeros(1,2);

x1 = Ex1l10m100n500train(:, 1:(end-1));
y1 = Ex1l10m100n500train(:, end);

testx = Ex1l10m100n500test(:, 1:(end-1));
testy = Ex1l10m100n500test(:, end);

mar = [2.0,0.3,0.04,0.006,0.001];
alpha = [1.1,1.01,1.005,1.0005,1.00001];
max_acc = -1;
best_margin = 0;
best_alpha = 0;

for win_m_rate = 1:numel(mar)
    for ap = 1:numel(alpha)
        [w] = winnow_margin(x1,y1,alpha(ap),mar(win_m_rate));
        accy = calaccuracy(testx,testy,w,-1*500);
        if(accy>max_acc)
            max_acc = accy;
            best_margin = mar(win_m_rate);
            best_alpha = alpha(ap);
        end
    end
end
winnow_m_max_accy_cache(1,1) = max_acc;
winnow_m_best_margin_cache(1,1) = best_margin;
winnow_m_best_alpha_cache(1,1) = best_alpha;


x2 = Ex1l10m100n1000train(:, 1:(end-1));
y2 = Ex1l10m100n1000train(:, end);

testx = Ex1l10m100n1000test(:, 1:(end-1));
testy = Ex1l10m100n1000test(:, end);

mar = [2.0,0.3,0.04,0.006,0.001];
alpha = [1.1,1.01,1.005,1.0005,1.00001];
max_acc = -1;
best_magin = 0;
best_alpha = 0;

for win_m_rate = 1:numel(mar)
    for ap = 1:numel(alpha)
        [w] = winnow_margin(x2,y2,alpha(ap),mar(win_m_rate));
        accy = calaccuracy(testx,testy,w,-1*1000);
        if(accy>max_acc)
            max_acc = accy;
            best_margin = mar(win_m_rate);
            best_alpha = alpha(ap);
        end
    end
end
winnow_m_max_accy_cache(1,2) = max_acc;
winnow_m_best_margin_cache(1,2) = best_margin;
winnow_m_best_alpha_cache(1,2) = best_alpha;

for disp_index = 1:2
    disp('For Winnow magin:')
    disp('Max accuracy:')
    disp( winnow_m_max_accy_cache(1,disp_index))
    disp('Learning Rate:')
    disp(winnow_m_best_alpha_cache(1,disp_index))
    disp('Margin:')
    disp(winnow_m_best_margin_cache(1,disp_index))
end 