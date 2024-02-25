% EE 456 HW3, Ian Deal, imd5205@psu.edu

clear all; % Clear all variables
clc; % Clear the command window
close all; % Close all figure windows

% load dataset 1
%load("Data\DataSet1_MP1.mat", "DataSet1");
%load("Data\DataSet1_MP1.mat", "DataSet1_targets");

% normalize DataSet1 such that it's value fall proportionally between 0 and 1
%DataSet1 = normalize(DataSet1);

% split dataset 1 into training and testing datasets
% elements of dataset 1 are chosen at random to be in training or testing
% the training set is 80% of dataset 1 while the test set is the remaining 20%

%rand_indices = randperm(6000, 4800); % choose 4800 (80% of 6000) random vectors
%S = DataSet1(rand_indices, :); % training set for dataset 1
%T = DataSet1_targets(rand_indices, :); % targets for dataset 1
%save("Data\Trainset_Dataset1.mat", "S");
%save("Data\Trainset_Dataset1_targets.mat", "T");

%remaining_indices = setdiff(1:6000, rand_indices); % the remaining 1200 (20% of 6000) vectors
%testset1 = DataSet1(remaining_indices, :); % testing set for dataset 1
%testset1_targets = DataSet1_targets(remaining_indices, :); % targets for the test set

%save("Data\Testset_Dataset1.mat", "testset1");
%save("Data\Testset_Dataset1_targets.mat", "testset1_targets");

load("Data\Trainset_Dataset1.mat", "S")
load("Data\Trainset_Dataset1_targets.mat", "T")
load("Data\Testset_Dataset1.mat", "testset1")
load("Data\Testset_Dataset1_targets.mat", "testset1_targets")

% init weight matrices to a zero mean distribution with a std = 1/sqrt(number of connections)
std1 = 1/sqrt(40);
W1 = std1 * rand(3,20); % init first weight matrix, weights from input to hidden layer
W1 = W1 - mean(W1(:)); % sets mean to zero

std2 = 1/sqrt(20);
W2 = std2 * rand(21,1); % init second weight matrix, weights from hidden layer to output
W2 = W2 - mean(W2(:));

% train the model using backprop
i = randi([1, 4800]);
iter_count = 0;
eta = 10^(-1); % learning rate
trainset_error = [];
testset_error = [];
threshold = 0;

while iter_count < 10000000
    x = S(i,:);
    [x, z, z_in, y, y_in] = forward_train(x,W1,W2); % initial forward pass
    
    % update output weights
    j = 2;
    sigma = (T(i)-y) * sech(y_in); % sech is the derivative of tanh
    while j <= 21
        delta = eta * sigma * z(j,:); %z(j,:)
        W2(j,:) = W2(j,:) + delta;
        j = j + 1;
    end
    W2(1,:) = W2(1,:) + (eta * sigma); % bias correction

    % update hidden weights
    j = 1;
    while j <= 20
        l = 2;
        while l <= 3
            sigma_in = sigma * W2(j,:);
            sigma_j = sigma_in * sech(z_in(j,:));
            delta = eta * sigma_j * x(l);
            W1(l,j) = W1(l,j) + delta;
            l = l + 1;
        end
        j = j + 1;
    end

    j = 1;
    while j <= 20
        sigma_in = sigma * W2(j,:);
        sigma_j = sigma_in * sech(1);
        W1(1,j) = W1(1,j) + (eta * sigma_j); % bias correction
        j = j + 1;
    end

    if mod(iter_count, 1000) == 0
        % plot the training and validation error for this epoch
        train_y = forward_test(S,W1,W2);
        test_y = forward_test(testset1,W1,W2);
        train_y(train_y >= threshold) = 1;
        train_y(train_y < threshold) = -1;
        test_y(test_y >= threshold) = 1;
        test_y(test_y < threshold) = -1;
        train_e = mean((train_y - T).^2);
        test_e = mean((test_y - testset1_targets).^2);
        trainset_error = [trainset_error train_e];
        testset_error = [testset_error test_e];
    end
    if eta > 10^(-5)
        eta = eta/1.08;
    end
    i = randi([1, 4800]);
    iter_count = iter_count + 1;
end

% Plot both vectors on the same plot
subplot(1,2,1);
plot(trainset_error, 'o-', 'LineWidth', 2, 'DisplayName', 'Training Error');
hold on;  % Hold the current plot so that the next plot is added to it
plot(testset_error, 's-', 'LineWidth', 2, 'DisplayName', 'Validation Error');

% Add labels and legend
xlabel('Epoch');
ylabel('Error');
title('Training and Validation Errors');
legend('show');

% Optionally, customize the plot further
hold off;  % Release the hold on the plot

subplot(1,2,2);
class1 = testset1(test_y == 1, :);
classneg1 = testset1(test_y == -1, :);
scatter(class1(:,1), class1(:,2), 'red', 'filled', 'DisplayName', 'Class 1');
hold on;
scatter(classneg1(:,1), classneg1(:,2), 'blue', 'filled', 'DisplayName', 'Class -1');
title('Plot of the Normalized Test Subest of Dataset1')

% forward pass functions for easier training/testing
function [x, z, z_in, y, y_in] = forward_train(x,weight1,weight2) % bipolar activation function
    x = [1 x];
    z_in = weight1'*x';
    z = [1; tanh(z_in)];
    y_in = z'*weight2;
    y = tanh(y_in);
end

function y = forward_test(x,weight1,weight2) % bipolar activation function
    bias = ones(size(x, 1),1);
    z = tanh(weight1'*[bias x]');
    y = tanh([bias'; z]'*weight2);
end

