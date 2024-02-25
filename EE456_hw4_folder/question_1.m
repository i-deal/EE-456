% EE 456 HW4, Ian Deal, imd5205@psu.edu

clear all; % Clear all variables
clc; % Clear the command window
close all; % Close all figure windows

% Question 1:
% Part a:

num_points = 1200;
x_range = [-0.6, 0.6];
y_range = [-0.6, 0.6];

% generate random x and y values within the specified ranges
x = (x_range(2) - x_range(1)) * rand(num_points, 1) + x_range(1);
y = (y_range(2) - y_range(1)) * rand(num_points, 1) + y_range(1);

data = [x, y];

% Part b, c, d:

W_range = [-1.2, 1.2];

% generate random weight values within the specified range
% a W vector of dimensions (2,576) was chosen so that the output would be a 24 * 24 lattice
% each (2,i) column vector represents the weight vector from the 2d input to the ith neuron in the lattice
W = (W_range(2) - W_range(1)) * rand(2, 576) + W_range(1);

% Part e:

% plot the data
figure(1)
disp('Plot data')
subplot(1,2,1);
scatter(x, y, 10, 'filled');
xlabel('X');
ylabel('Y');
title('Random Distribution of (X, Y) Pairs');

% Part f:

% plot initial weight values
disp('Plot initial weight values')
subplot(1,2,2);
scatter(W(1,:), W(2,:), 10, 'filled');
xlabel('X');
ylabel('Y');
title('Initial Distribution of Weight Values');
% connect the points with lines using the plot function
hold on;
for i = 1:575
    plot([W(1,i), W(1,i+1)], [W(2,i), W(2,i+1)], 'Color', 'b');
end
i=575;
plot([W(1,i), W(1,1)], [W(2,i), W(2,1)], 'Color', 'b');
hold off;

drawnow();

% Part g:

W = W';
alpha = 0.9; % starting learning rate
R = 576; % starting neighborhood rate
iter_count = 0;

% the map starts with a high learning rate and a large R to force the weights
% to 'point' towards a given data point, this helps the map to learn course features
% of the data very quickly and then as the learning rate and R decrease, the map
% learns finer features over many more iterations.
while iter_count < 300000
    idx = randi([1, 1200]); % choose a random sample from the dataset
    x = data(idx,:);
    min_idx = 1;
    min = norm(W(1,:)-data(1,:))^2; % compute an initial minimum distance for the chosen datapoint

    % compute the W index with the smallest distance from the sample point
    for i = 1:576
        cur = norm(W(i,:) - x)^2;
        if cur < min
            min = cur;
            min_idx = i;
        end
    end

    % determine the neighborhood around the winning neuron
    l_idx = int32(min_idx - R);
    r_idx = int32(min_idx + R);
    if l_idx < 1
        l_idx = 1;
    end

    if r_idx > 576
        r_idx = 576;
    end

    % update all weights within the neighborhood centered around the winning neuron
    for i = l_idx:r_idx
        w = W(i,:);
        W(i,:) = w + alpha*(x - w);
    end

    if alpha > 0.01
        alpha = (alpha)^(1/3); % decrease alpha by taking the cube root
    end

    if R > 0
        R = 0.7*R; % decrease the neighborhood size by thirty percent each iteration
    end
    
    if iter_count == 50000
        W1 = W; % save the ordering phase weights after 50k iterations
    end
    iter_count = iter_count + 1;
end

% plot the saved ordering phase weight matrix
figure(2)
disp('Plot Ordering Phase')
W1 = W1';
subplot(1,2,1);
scatter(W1(1,:), W1(2,:), 10, 'filled');
xlabel('X');
ylabel('Y');
title('Ordering Phase (50k iterations)');
% connect the points with lines using the plot function
hold on;
for i = 1:575
     plot([W1(1,i), W1(1,i+1)], [W1(2,i), W1(2,i+1)], 'Color', 'b');
end
i=575;
plot([W1(1,i), W1(1,1)], [W1(2,i), W1(2,1)], 'Color', 'b');
hold off;

% Part h:

% plot the final weight matrix: convergence phase
W = W';
disp('Plot Convergence Phase')
subplot(1,2,2);
scatter(W(1,:), W(2,:), 10, 'filled');
xlabel('X');
ylabel('Y');
title('Convergence Phase (300k iterations)');
% connect the points with lines using the plot function
hold on;
for i = 1:575
    plot([W(1,i), W(1,i+1)], [W(2,i), W(2,i+1)], 'Color', 'b');
end
i=575;
plot([W(1,i), W(1,1)], [W(2,i), W(2,1)], 'Color', 'b');
hold off;

