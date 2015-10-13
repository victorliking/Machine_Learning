
l = 10;
n = 1000;
m = [100;500;1000];
load ./data/ex3data.mat;
perceptron_margin_max_acc = zeros(1,numel(m));
per_best_rate_cache = zeros(1,numel(m));

winnow_max_acc = zeros(1,numel(m));
winnow_best_rate_cache = zeros(1,numel(m));

winnow_margin_max_acc = zeros(1,numel(m));
winnow_m_best_alpha_cache = zeros(1,numel(m));
winnow_m_best_margin_cache = zeros(1,numel(m));

adagrad_max_acc = zeros(1,numel(m));
adagrad_best_rate_cache = zeros(1,numel(m));

for m_index = 1:numel(m)
    if m_index == 1
        trainx = Ex3l10m100n1000randomtrain(:, 1:(end-1));
        trainy = Ex3l10m100n1000randomtrain(:, end);
        testx = Ex3l10m100n1000randomtest(:, 1:(end-1));
        testy = Ex3l10m100n1000randomtest(:, end);
    elseif m_index == 2
        trainx = Ex3l10m500n1000randomtrain(:, 1:(end-1));
        trainy = Ex3l10m500n1000randomtrain(:, end);
        testx = Ex3l10m500n1000randomtest(:, 1:(end-1));
        testy = Ex3l10m500n1000randomtest(:, end);
    elseif m_index == 3
        trainx = Ex3l10m1000n1000randomtrain(:, 1:(end-1));
        trainy = Ex3l10m1000n1000randomtrain(:, end);
        testx = Ex3l10m1000n1000randomtest(:, 1:(end-1));
        testy = Ex3l10m1000n1000randomtest(:, end);
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
    perceptron_margin_max_acc(1,m_index) = max_accuracy;
    per_best_rate_cache(1,m_index) = learning_rate;
    
   %winnow
    alpha = [1.1, 1.01, 1.005, 1.0005, 1.0001];    
    max_accuracy = -1;
    best_alpha = 0;
    for idx = 1:numel(alpha)
        [weights] = winnow(trainx,trainy,alpha(idx));
        accuracy_pc = calaccuracy(testx,testy,weights,-1*n);
        if(accuracy_pc > max_accuracy)
            max_accuracy = accuracy_pc;
            best_alpha = alpha(idx);
        end
    end
    winnow_max_acc(1,m_index) = max_accuracy;
    winnow_best_rate_cache(1,m_index) = best_alpha;
    
    % For Winnow with Margin
    margins = [2.0, 0.3, 0.04, 0.006, 0.001];
    alpha = [1.1, 1.01, 1.005, 1.0005, 1.0001];    
    max_accuracy = -1;
    best_alpha = 0;
    margin_val = 0;
    for idx = 1:numel(alpha)
        for idy = 1:numel(margins)
            [weights] = winnow_margin(trainx,trainy,alpha(idx),margins(idy));
            accuracy_pc = calaccuracy(testx,testy,weights,-1*n);
            if(accuracy_pc > max_accuracy)
                max_accuracy = accuracy_pc;
                best_alpha = alpha(idx);
                margin_val = margins(idy);
            end 
        end
    end  
    winnow_margin_max_acc(1,m_index) = max_accuracy;
    winnow_m_best_alpha_cache(1,m_index) = best_alpha;
    winnow_m_best_margin_cache(1,m_index) = margin_val;
    
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
    adagrad_max_acc(1,m_index) = max_accuracy;
    adagrad_best_rate_cache(1,m_index) = learning_rate;
    
end


for m_index = 1:numel(m)
    if m_index == 1
        train_x = Ex3l10m100n1000train(:, 1:(end-1));
        train_y = Ex3l10m100n1000train(:, end);
        test_x = Ex3l10m100n1000test(:, 1:(end-1));
        test_y = Ex3l10m100n1000test(:, end);
    elseif m_index == 2
        train_x = Ex3l10m500n1000train(:, 1:(end-1));
        train_y = Ex3l10m500n1000train(:, end);
        test_x = Ex3l10m500n1000test(:, 1:(end-1));
        test_y = Ex3l10m500n1000test(:, end);
    elseif m_index == 3
        train_x = Ex3l10m1000n1000train(:, 1:(end-1));
        train_y = Ex3l10m1000n1000train(:, end);
        test_x = Ex3l10m1000n1000test(:, 1:(end-1));
        test_y = Ex3l10m1000n1000test(:, end);
    end
    
    % For perceptron.
    [weights,theta] = perceptron(train_x,train_y);
    perceptron_accy(m_index) = calaccuracy(test_x,test_y,weights,theta);
    
    % For perceptron with margin 
    [weights,theta] = perceptron_margin(train_x,train_y,per_best_rate_cache(1,m_index));
    perceptron_m_accy(m_index) = calaccuracy(test_x,test_y,weights,theta);
          
    % For Winnow 
    [weights] = winnow(train_x,train_y,winnow_best_rate_cache(1,m_index));
    winnow_accy(m_index) = calaccuracy(test_x,test_y,weights,-1*n);
        
    % For Winnow with Margin
    [weights] = winnow_margin(train_x,train_y,winnow_m_best_alpha_cache(1,m_index),winnow_m_best_margin_cache(1,m_index));
    winnow_m_accy(m_index) = calaccuracy(test_x,test_y,weights,-1*n);
      
    % For Adagrad.
    [weights,theta] = adagrad(train_x,train_y,adagrad_best_rate_cache(1,m_index));
    adagrad_accy(m_index) = calaccuracy(test_x,test_y,weights,theta);
    
end



fileID = fopen('ex3_result.txt','w');


%result for perceptron
fprintf(fileID,'the data for perceptron \n');
fprintf(fileID,'accuracy: ');
fprintf(fileID,'%f ',perceptron_accy);
fprintf(fileID,'\n');

%result for perceptron margin
fprintf(fileID,'the data for perceptron margin\n');
fprintf(fileID,'accuracy: ');
fprintf(fileID,'%f ',perceptron_m_accy);
fprintf(fileID,'\n');
fprintf(fileID,'best rate: ');
fprintf(fileID,'%f ',per_best_rate_cache);
fprintf(fileID,'\n');


%result for winnow
fprintf(fileID,'the data for winnow\n');
fprintf(fileID,'accuracy: ');
fprintf(fileID,'%f ',winnow_accy);
fprintf(fileID,'\n');
fprintf(fileID,'best rate: ');
fprintf(fileID,'%f ',winnow_best_rate_cache);
fprintf(fileID,'\n');

%result for winnow margin
fprintf(fileID,'the data for winnow margin\n');
fprintf(fileID,'accuracy: ');
fprintf(fileID,'%f ',winnow_m_accy);
fprintf(fileID,'\n');
fprintf(fileID,'best margin: ');
fprintf(fileID,'%f ',winnow_m_best_margin_cache);
fprintf(fileID,'\n');
fprintf(fileID,'best alpha: ');
fprintf(fileID,'%f ',winnow_m_best_alpha_cache);
fprintf(fileID,'\n');
%sult for adagrad
fprintf(fileID,'the data for adagrad\n');
fprintf(fileID,'max accuracy: ');
fprintf(fileID,'%f ',adagrad_accy);
fprintf(fileID,'\n');
fprintf(fileID,'best rate: ');
fprintf(fileID,'%f ',adagrad_best_rate_cache);
fprintf(fileID,'\n');

fclose(fileID);