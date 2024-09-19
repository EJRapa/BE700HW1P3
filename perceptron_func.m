%% Set up data
clear, close all

HW1_sortdat = readtable("..\num_2\BE700-HW1\sorted_HW1_dat.csv");
HW1_sortdat = cat(2, ...
    ones(size(HW1_sortdat,1),1), ...
    table2array(HW1_sortdat(:,1:end-1)), ...
    strcmp(HW1_sortdat.prognosis,'GOOD'));

%% Set up variables

% For batch method, find the average of the modified weights and change
% according to that

%% Constants

ER_LIM = 0;
EPOC_LIM = 2*10^4;
ALPHA = 0.5:5:20;
DIM = 30;

% For the additional dimension to allow for n points in n dimensions

assert(DIM < 31);
DIM = DIM + 1;

for i=1:length(ALPHA)
HW1_sortdat_train = cat(2,HW1_sortdat(:,1:DIM),...
    HW1_sortdat(:,end));

%% Adjust input wieghts during a certain number of epochs

[row, col] = size(HW1_sortdat_train);
weights = 2.*rand(col-1,1) - 1;
epoch = 0;

t_accur = zeros(1,EPOC_LIM);
wgt = zeros(5,EPOC_LIM);

tic
while true

    [w_err, weights] = batch_learn(HW1_sortdat_train,weights,ALPHA(i));
        
    % Store training accuracy at certain epoch
    epoch = epoch + 1;

    wgt(:,epoch) = weights(1:5,:);
    t_accur(epoch) = 100-(w_err/row)*100;

    if (w_err <= ER_LIM || epoch == EPOC_LIM) break; end
end
toc

plot(1:epoch,t_accur(1:epoch),LineWidth=2);
title(sprintf("Classification Accuracy for Training Data After n Epochs with %d Dimensions",DIM-1));

xlabel("Epochs");
ylabel("Classification Accuracy");
legend(sprintf("alpha = %.1f",ALPHA(i)));

hold on
end

% Legend Concatenation
Legend=cell(length(ALPHA),1);
 for iter=1:length(ALPHA)
   Legend{iter}=strcat('alpha =  ', num2str(ALPHA(iter)));
 end
 legend(Legend)

%% Testing Set
HW1_sortdat_test = cat(2,HW1_sortdat(46:end,1:DIM),...
    HW1_sortdat(46:end,end));

y = sum(HW1_sortdat_test(:,1:end-1) .* weights.',2) >= 0;

corct = sum(y == HW1_sortdat_test(:,end));

accuracy = (corct/size(HW1_sortdat_test,1))*100;

fprintf("Accuracy is %.2f\n",accuracy);

%% Plot weights 

figure 

for i=1:5
    plot(1:epoch,wgt(i,1:epoch));
    hold on
end

title("Change in Weights over n Epochs");
xlabel("Epochs");
ylabel("Weight Score");
legend("w1","w2","w3","w4","w5");