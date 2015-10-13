load ./data/ex2data.mat;
load ./data/ex2.mat
l = 10;
m = 20;    
n = 40:40:200;


winnow_m_max_accy_cache = zeros(1,numel(n));
winnow_m_best_margin_cache = zeros(1,numel(n));
winnow_m_best_alpha_cache = zeros(1,numel(n));


for n_index = 1:numel(n)
    if n_index == 1
        trainx = Ex2l10m20n40train(:, 1:(end-1));
        trainy = Ex2l10m20n40train(:, end);
        testx = Ex2l10m20n40test(:, 1:(end-1));
        testy = Ex2l10m20n40test(:, end);
    elseif n_index == 2
        trainx = Ex2l10m20n80train(:, 1:(end-1));
        trainy = Ex2l10m20n80train(:, end);
        testx = Ex2l10m20n80test(:, 1:(end-1));
        testy = Ex2l10m20n80test(:, end);
    elseif n_index == 3
        trainx = Ex2l10m20n120train(:, 1:(end-1));
        trainy = Ex2l10m20n120train(:, end);
        testx = Ex2l10m20n120test(:, 1:(end-1));
        testy = Ex2l10m20n120test(:, end);
    elseif n_index == 4
        trainx = Ex2l10m20n160train(:, 1:(end-1));
        trainy = Ex2l10m20n160train(:, end);
        testx = Ex2l10m20n160test(:, 1:(end-1));
        testy = Ex2l10m20n160test(:, end);
    elseif n_index == 5
        trainx = Ex2l10m20n200train(:, 1:(end-1));
        trainy = Ex2l10m20n200train(:, end);
        testx = Ex2l10m20n200test(:, 1:(end-1));
        testy = Ex2l10m20n200test(:, end);
    end
    
    margins = [2.0, 0.3, 0.04, 0.006, 0.001];
    alpha = [1.1, 1.01, 1.005, 1.0005, 1.0001];    
    max_accuracy = -1;
    best_alpha = 0;
    margin_val = 0;
    for idx = 1:numel(alpha)
        for idy = 1:numel(margins)
            [weights] = winnow_margin(trainx,trainy,alpha(idx),margins(idy));
            accuracy_pc = calaccuracy(testx,testy,weights,-1*n(n_index));
            if(accuracy_pc > max_accuracy)
                max_accuracy = accuracy_pc;
                best_alpha = alpha(idx);
                margin_val = margins(idy);
            end 
        end
    end  
    winnow_m_max_accy_cache(1,n_index) = max_accuracy;
    winnow_m_best_alpha_cache(1,n_index) = best_alpha;
    winnow_m_best_margin_cache(1,n_index) = margin_val;
    
end
for disp_index = 1:numel(n)
    disp('For Winnow Margin:')
    disp('Max accuracy:')
    disp(winnow_m_max_accy_cache(1,disp_index))
    disp('margin Rate:')
    disp(winnow_m_best_margin_cache(1,disp_index))
    disp('alpha:')
    disp(winnow_m_best_alpha_cache(1,disp_index))
end 