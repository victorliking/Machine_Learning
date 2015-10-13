load ./data/ex1data.mat

per_max_accy_cache = zeros(1,2);
per_best_rate_cache = zeros(1,2);

winnow_max_accy_cache = zeros(1,2);
winnow_best_rate_cache = zeros(1,2);

winnow_m_max_accy_cache = zeros(1,2);
winnow_m_best_margin_cache = zeros(1,2);
winnow_m_best_alpha_cache = zeros(1,2);

adagrad_max_accy_cache = zeros(1,2);
adagrad_best_rate_cache = zeros(1,2);

x1 = Ex1l10m100n500train(:, 1:(end-1));
y1 = Ex1l10m100n500train(:, end);

testx = Ex1l10m100n500test(:, 1:(end-1));
testy = Ex1l10m100n500test(:, end);

%test perceptron margin para
rates = [1.5,0.25,0.03,0.05,0.001];
max_acc = -1;
rate = 0;

for per_rate = 1:numel(rates)
    [w,theta] = perceptron_margin(x1,y1,rates(per_rate));
    accy = calaccuracy(testx,testy,w,theta);
    if(accy>max_acc)
        max_acc = accy;
        rate = rates(per_rate);
    end
end
per_max_accy_cache(1,1) = max_acc;
per_best_rate_cache(1,1) = rate;

%test winnow para
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


%test winnow margin para
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


%test adagrad
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

%test perceptron margin para
rates = [1.5,0.25,0.03,0.05,0.001];
max_acc = -1;
rate = 0;

for per_rate = 1:numel(rates)
    [w,theta] = perceptron_margin(x2,y2,rates(per_rate));
    accy = calaccuracy(testx,testy,w,theta);
    if(accy>max_acc)
        max_acc = accy;
        rate = rates(per_rate);
    end
end
per_max_accy_cache(1,2) = max_acc;
per_best_rate_cache(1,2) = rate;

%test winnow para
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


%test winnow margin para
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


%test adagrad
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



fileID = fopen('ex1_result.txt','w');

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

l = 10;
m = 100; 
n = [500;1000];
%plot part
for index = 1:numel(n)
    [y,x] = gen(l,m,n(index),50000,0);
    per_mis = perceptron_mistake(x,y);
    per_m_mis = perceptron_margin_mistake(x,y,per_best_rate_cache(1,index));
    win_mis =  winnow_mistake(x,y,winnow_best_rate_cache(1,index));
    win_m_mis = winnow_margin_mistake(x,y,winnow_m_best_alpha_cache(1,index),winnow_m_best_margin_cache(1,index));
    ada_mis = adagrad_mistake(x,y,adagrad_best_rate_cache(index));
    
    xaxis = 1:100:50000;
    x = [0 xaxis];
    y_per = [0 per_mis];
    y_per_m = [0 per_m_mis];
    y_win = [0 win_mis];
    y_win_m = [0 win_m_mis];
    y_ada = [0 ada_mis];
    
    figure;
    plot(x,y_per,'r',x,y_per_m,'g',x,y_win,'b',x,y_win_m,'y',x,y_ada,'m')
    xlabel('Number of Examples N');
    ylabel('Number of Mistakes W');
    str = sprintf('N vs. W for n=%d',n(index));
    title(str);
    legend('Perceptron','Perceptron With Margin','Winnow','Winnow With Margin','Adagrad');
end

