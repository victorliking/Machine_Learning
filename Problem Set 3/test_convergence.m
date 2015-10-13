l = 10;
m = 20;    
n = 40:40:200;


winnow_m_max_accy_cache = zeros(1,numel(n));
winnow_m_best_margin_cache = zeros(1,numel(n));
winnow_m_best_alpha_cache = zeros(1,numel(n));


for n_index = 1:numel(n)
    [y,x] = gen(l,m,n(n_index),50000,0);
    %10% data is training set.
    d_1_x = x(1:5000,:);
    d_1_y = y(1:5000,:);
    %10% data is testing set.
    d_2_x = x(5001:10000,:);
    d_2_y = y(5001:10000,:);
    margins = [2.0, 0.3, 0.04, 0.006, 0.001];
    alpha = [1.1, 1.01, 1.005, 1.0005, 1.0001];    
    max_accuracy = -1;
    prom_demot_val = 0;
    margin_val = 0;
    for idx = 1:numel(alpha)
        for idy = 1:numel(margins)
            [weights] = winnow_margin(d_1_x,d_1_y,alpha(idx),margins(idy));
            accuracy_pc = calaccuracy(d_2_x,d_2_y,weights,-1*n(n_index));
            if(accuracy_pc > max_accuracy)
                max_accuracy = accuracy_pc;
                prom_demot_val = alpha(idx);
                margin_val = margins(idy);
            end 
        end
    end  
    winnow_m_max_accy_cache(1,n_index) = max_accuracy;
    winnow_m_best_alpha_cache(1,n_index) = prom_demot_val;
    winnow_m_best_margin_cache(1,n_index) = margin_val;
end 

for disp_index = 1:numel(n)
    disp('For Adagrad:')
    disp('Max accuracy:')
    disp(winnow_m_max_accy_cache(1,disp_index))
    disp('margin Rate:')
    disp(winnow_m_best_margin_cache(1,disp_index))
    disp('alpha:')
    disp(winnow_m_best_alpha_cache(1,disp_index))
end 