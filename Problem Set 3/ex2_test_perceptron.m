load ./data/ex2data.mat;
load ./data/ex2.mat
l = 10;
m = 20;    
n = 40:40:200;

per_max_accy_cache = zeros(1,numel(n));
per_best_rate_cache = zeros(1,numel(n));

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

    % For perceptron with margin (Tune Learning Rate):
    rates = [1.5, 0.25, 0.03, 0.005, 0.001];
    max_accuracy = -1;
    learning_rate = 0;
    for idx = 1:numel(rates)
        [weights,theta] = perceptron_margin(trainx,trainy,rates(idx));
        accuracy_pc = calaccuracy(testx,testy,weights,theta);
        if(accuracy_pc > max_accuracy)
            max_accuracy = accuracy_pc;
            learning_rate = rates(idx);
        end
    end
    per_max_accy_cache(1,n_index) = max_accuracy;
    per_best_rate_cache(1,n_index) = learning_rate;
end


for disp_index = 1:numel(n)
    disp('For Perceptron:')
    disp('Max accuracy:')
    disp( per_max_accy_cache(1,disp_index))
    disp('margin Rate:')
    disp(per_best_rate_cache(1,disp_index))
end 