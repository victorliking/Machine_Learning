l = 10;
n = 1000;
m = [100;500;1000];

percerton_m_m_max_acc_cache = zeros(1,numel(m));
percerton_m_m_rate_cache = zeros(1,numel(m));

percerton_m_max_acc_cache = zeros(1,numel(m));
percerton_m_rate_cache = zeros(1,numel(m));

load ./data/ex4data.mat
load ./data/ex4.mat

for index = 1:numel(m)
    if index == 1
        trainx = Ex4l10m100n1000randomtrain(:, 1:(end-1));
        trainy = Ex4l10m100n1000randomtrain(:, end);
        testx = Ex4l10m100n1000randomtest(:, 1:(end-1));
        testy = Ex4l10m100n1000randomtest(:, end);
    elseif index == 2
        trainx = Ex4l10m500n1000randomtrain(:, 1:(end-1));
        trainy = Ex4l10m500n1000randomtrain(:, end);
        testx = Ex4l10m500n1000randomtest(:, 1:(end-1));
        testy = Ex4l10m500n1000randomtest(:, end);
    elseif index == 3
        trainx = Ex4l10m1000n1000randomtrain(:, 1:(end-1));
        trainy = Ex4l10m1000n1000randomtrain(:, end);
        testx = Ex4l10m1000n1000randomtest(:, 1:(end-1));
        testy = Ex4l10m1000n1000randomtest(:, end);
    end

    % For Modified perceptron with margin (Tune Learning Rate):
    rates = [1.5, 0.25, 0.03, 0.005, 0.001];
    max_accuracy = -1;
    learning_rate = 0;
    for idx = 1:numel(rates)
        [weights,theta] = perceptron_margin_modified(trainx,trainy,rates(idx));
        accuracy_pc = calaccuracy(testx,testy,weights,theta);
        if(accuracy_pc > max_accuracy)
            max_accuracy = accuracy_pc;
            learning_rate = rates(idx);
        end
    end
    percerton_m_m_max_acc_cache(1,index) = max_accuracy;
    percerton_m_m_rate_cache(1,index) = learning_rate;
        
    % For original perceptron with margin (Tune Learning Rate):
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
    percerton_m_max_acc_cache(1,index) = max_accuracy;
    percerton_m_rate_cache(1,index) = learning_rate;
end

for m_index = 1:numel(m)
    if m_index == 1
        train_x = Ex4l10m100n1000train(:, 1:(end-1));
        train_y = Ex4l10m100n1000train(:, end);
        test_x = Ex4l10m100n1000test(:, 1:(end-1));
        test_y = Ex4l10m100n1000test(:, end);
    elseif m_index == 2
        train_x = Ex4l10m500n1000train(:, 1:(end-1));
        train_y = Ex4l10m500n1000train(:, end);
        test_x = Ex4l10m500n1000test(:, 1:(end-1));
        test_y = Ex4l10m500n1000test(:, end);
    elseif m_index == 3
        trainx = Ex4l10m1000n1000train(:, 1:(end-1));
        trainy = Ex4l10m1000n1000train(:, end);
        testx = Ex4l10m1000n1000test(:, 1:(end-1));
        testy = Ex4l10m1000n1000test(:, end);
    end

    % For Modified perceptron with margin (With the best learning rate):
    [weights,theta] = perceptron_margin_modified(train_x,train_y,percerton_m_m_rate_cache(1,m_index));
    mod_acc(m_index) = calaccuracy(test_x,test_y,weights,theta);
    
    % For the original perceptron with margin (With the best learning rate):
    [weights,theta] = perceptron_margin(train_x,train_y,percerton_m_rate_cache(1,m_index));
    acc(m_index) = calaccuracy(test_x,test_y,weights,theta);
end

fileID = fopen('ex4_result.txt','w');


%result for modified
fprintf(fileID,'the data for modified \n');
fprintf(fileID,'accuracy: ');
fprintf(fileID,'%f ',mod_acc);
fprintf(fileID,'\n');
fprintf(fileID,'modified best rate: ');
fprintf(fileID,'%f ',percerton_m_m_rate_cache);
fprintf(fileID,'\n');


%result for normal
fprintf(fileID,'the data for normal \n');
fprintf(fileID,'accuracy: ');
fprintf(fileID,'%f ',acc);
fprintf(fileID,'\n');
fprintf(fileID,'best rate: ');
fprintf(fileID,'%f ',percerton_m_rate_cache);
fprintf(fileID,'\n');

fclose(fileID);