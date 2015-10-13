load ./data/ex2data.mat;
load ./data/ex2.mat
l = 10;
m = 20;    
n = 40:40:200;

per_max_accy_cache = zeros(1,numel(n));
per_best_rate_cache = zeros(1,numel(n));

winnow_max_accy_cache = zeros(1,numel(n));
winnow_best_rate_cache = zeros(1,numel(n));

winnow_m_max_accy_cache = zeros(1,numel(n));
winnow_m_best_margin_cache = zeros(1,numel(n));
winnow_m_best_alpha_cache = zeros(1,numel(n));

adagrad_max_accy_cache = zeros(1,numel(n));
adagrad_best_rate_cache = zeros(1,numel(n));

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

    % For perceptron with margin 
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
    
    % For Winnow 
    alpha = [1.1, 1.01, 1.005, 1.0005, 1.0001];    
    max_accuracy = -1;
    best_alpha = 0;
    for idx = 1:numel(alpha)
        [weights] = winnow(trainx,trainy,alpha(idx));
        accuracy_pc = calaccuracy(testx,testy,weights,-1*n(n_index));
        if(accuracy_pc > max_accuracy)
            max_accuracy = accuracy_pc;
            best_alpha = alpha(idx);
        end
    end
    winnow_max_accy_cache(1,n_index) = max_accuracy;
    winnow_best_rate_cache(1,n_index) = best_alpha;
    
    % For Winnow with Margin 
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
    
    % For Adagrad. 
    rates = [1.5, 0.25, 0.03, 0.005, 0.001];
    max_accuracy = -1;
    learning_rate = 0;
    for idx = 1:numel(rates)
        [weights,theta] = adagrad(trainx,trainy,rates(idx));
        accuracy_pc = calaccuracy(testx,testy,weights,theta);
        if(accuracy_pc > max_accuracy)
            max_accuracy = accuracy_pc;
            learning_rate = rates(idx);
        end
    end
    adagrad_max_accy_cache(1,n_index) = max_accuracy;
    adagrad_best_rate_cache(1,n_index) = learning_rate;
end

fileID = fopen('ex2_result.txt','w');

%result for perceptron

fprintf(fileID,'the data for perceptron margin\n');
fprintf(fileID,'max accuracy: ');
fprintf(fileID,'%f ',per_max_accy_cache);
fprintf(fileID,'\n');
fprintf(fileID,'best rate: ');
fprintf(fileID,'%f ',per_best_rate_cache);
fprintf(fileID,'\n');

%result for winnow
fprintf(fileID,'the data for winnow\n');
fprintf(fileID,'max accuracy: ');
fprintf(fileID,'%f ',winnow_max_accy_cache);
fprintf(fileID,'\n');
fprintf(fileID,'best rate: ');
fprintf(fileID,'%f ',winnow_best_rate_cache);
fprintf(fileID,'\n');

%result for winnow margin
fprintf(fileID,'the data for winnow margin\n');
fprintf(fileID,'max accuracy: ');
fprintf(fileID,'%f ',winnow_m_max_accy_cache);
fprintf(fileID,'\n');
fprintf(fileID,'best margin: ');
fprintf(fileID,'%f ',winnow_m_best_margin_cache);
fprintf(fileID,'\n');
fprintf(fileID,'best alpha: ');
fprintf(fileID,'%f ',winnow_m_best_alpha_cache);
fprintf(fileID,'\n');

%result for adagrad
fprintf(fileID,'the data for adagrad\n');
fprintf(fileID,'max accuracy: ');
fprintf(fileID,'%f ',adagrad_max_accy_cache);
fprintf(fileID,'\n');
fprintf(fileID,'best rate: ');
fprintf(fileID,'%f ',adagrad_best_rate_cache);
fprintf(fileID,'\n');

fclose(fileID);


for test_index = 1:numel(n)
        
    if test_index == 1
        x = Ex2l10m20n40(:, 1:(end-1));
        y = Ex2l10m20n40(:, end);
    elseif test_index == 2
        x = Ex2l10m20n80(:, 1:(end-1));
        y = Ex2l10m20n80(:, end);
    elseif test_index == 3
        x = Ex2l10m20n120(:, 1:(end-1));
        y = Ex2l10m20n120(:, end);
    elseif test_index == 4
        x = Ex2l10m20n160(:, 1:(end-1));
        y = Ex2l10m20n160(:, end);
    elseif test_index == 5
        x = Ex2l10m20n200(:, 1:(end-1));
        y = Ex2l10m20n200(:, end);
    end
    mistakes_perceptron(test_index) = perceptron_convergence(x,y);
    mistakes_perceptron_margin(test_index) = perceptron_margin_convergence(x,y,per_best_rate_cache(1,test_index));
    mistakes_winnow(test_index) = winnow_convergence(x,y,winnow_best_rate_cache(1,test_index));
    mistakes_winnow_margin(test_index) = winnow_margin_convergence(x,y,winnow_m_best_alpha_cache(1,test_index), winnow_m_best_margin_cache(1,test_index));
    mistakes_adagrad(test_index) = adagrad_convergence(x,y,adagrad_best_rate_cache(1,test_index));

end

%Plot for each algorithm W vs. n
x = [0 n];
yp = [0 mistakes_perceptron];
ypm = [0 mistakes_perceptron_margin];
yw = [0 mistakes_winnow];
ywm = [0 mistakes_winnow_margin];
ya = [0 mistakes_adagrad];
   
figure;
plot(x,yp,'r',x,ypm,'g',x,yw,'b',x,ywm,'y',x,ya,'k')
xlabel('Values n');
ylabel('Number of Mistakes W');
title('W vs. n ');
legend('Perceptron','Perceptron With Margin','Winnow','Winnow With Margin','Adagrad');
