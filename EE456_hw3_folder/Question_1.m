% EE 456 HW3, Ian Deal, imd5205@psu.edu

clear all; % Clear all variables
clc; % Clear the command window
close all; % Close all figure windows

question_1 = 1;

% run the first question
if question_1 == 1
    disp('question 1:')
    % part a:
    disp('part a:')

    % load the data from the .mat file as variables S and T which are 9,7,3 matrices
    load("Data\SandT_patterns_HW3_P1.mat", "S");
    load("Data\SandT_patterns_HW3_P1.mat", "T");
    S = reshape(S, 63, 3); % flattens the S matrix for easier forward pass computing
    T = reshape(T, 63, 3); % flattens the T matrix for easier forward pass computing
    W = zeros(63, 63); % init weight matrix
    i = 1;

    % training weights using the Hebb rule
    while i < length(S(1,:)) + 1
        x = S(:,i); % the ith sample image
        j = 1;
        % Hebb rule, update the weight vector W_j by adding the product of x_j and T_i
        while j < 64
            delta = x(j) * T(:,i); % x_j times the target of T_i
            W(:,j) = W(:,j) + delta; % update the weight vector, W_j
            j = j + 1;
        end
        i = i + 1;
    end
    
    i = 1;
    % testing loop
    while i < length(S(1,:)) + 1
        out = S(:,i)' * W'; % pass each input through the network be computing the outer product with the weight matrix
        act = arrayfun(@bipolar_activation, out); % apply the bipolar activation function
        disp('Input:')
        disp(arrayfun(@vis, reshape(S(:,i), 9, 7, 1)));
        disp('Target:');
        disp(arrayfun(@vis, reshape(T(:,i), 9, 7, 1)));
        disp('Output:');
        disp(arrayfun(@vis, reshape(act, 9, 7, 1)));
        disp(isequal(reshape(act, 9, 7, 1), reshape(T(:,i), 9, 7, 1))); % checks whether the output exactly matches the target
        i = i + 1;
    end

    % part b:
    disp('part b:')

    load("Data\SandT_patterns_HW3_P1.mat", "S");
    load("Data\SandT_patterns_HW3_P1.mat", "T");
    
    % s4 and s5 are the new sample input images and t4 and t5 are the corresponding target images
    s4 = [-1 1 1 1 1 1 -1; -1 1 -1 -1 -1 1 -1; -1 1 -1 -1 -1 1 -1; 1 -1 1 -1 1 -1 1; -1 1 -1 -1 -1 1 -1; 1 -1 1 -1 1 -1 1; -1 1 -1 -1 -1 1 -1; -1 1 -1 -1 -1 1 -1; -1 1 1 1 1 1 -1;];
    s5 = [-1 -1 -1 1 -1 -1 -1; 1 -1 1 1 1 -1 1; 1 -1 1 1 1 -1 1; -1 -1 -1 1 -1 -1 -1; 1 1 -1 -1 -1 1 1; -1 -1 -1 1 -1 -1 -1; 1 -1 1 1 1 -1 1; 1 -1 1 1 1 -1 1; -1 -1 -1 1 -1 -1 -1;];
    S1 = cat(3, s4, s5); % concat the new samples together 
    S = cat(3, S, S1);  % concat the new samples onto the existing samples
    
    t4 = [-1 -1 -1 -1 -1 -1 -1; -1 1 1 1 1 1 -1; 1 -1 -1 -1 -1 -1 1; 1 -1 1 1 1 -1 1; 1 -1 1 -1 1 -1 1; 1 -1 1 1 1 -1 1; 1 -1 -1 -1 -1 -1 1; -1 1 1 1 1 1 -1; -1 -1 -1 -1 -1 -1 -1;];
    t5 = [-1 1 1 1 1 1 -1; 1 -1 -1 -1 -1 -1 1; -1 1 1 1 1 1 -1; -1 -1 -1 -1 -1 -1 -1; -1 -1 -1 -1 -1 -1 -1; -1 -1 -1 -1 -1 -1 -1; -1 1 1 1 1 1 -1; 1 -1 -1 -1 -1 -1 1; -1 1 1 1 1 1 -1;];
    T1 = cat(3, t4, t5); % concat the new targets together 
    T = cat(3, T, T1); % concat the new targets onto the existing targets

    S = reshape(S, 63, 5); % flattens the S matrix for easier forward pass computing
    T = reshape(T, 63, 5); % flattens the T matrix for easier forward pass computing
    W = zeros(63, 63); % init weight matrix
    
    % training weights using the Hebb rule
    i = 1;
    while i < length(S(1,:)) + 1
        x = S(:,i); % the ith sample image
        j = 1;
        % Hebb rule, update the weight vector W_j by adding the product of x_j and T_i
        while j < 64
            delta = x(j) * T(:,i); % x_j times the target of T_i
            W(:,j) = W(:,j) + delta; % update the weight vector, W_j
            j = j + 1;
        end
        i = i + 1;
    end
    
    % testing loop
    i = 1;
    while i < length(S(1,:)) + 1
        out = S(:,i)' * W'; % pass each input through the network be computing the outer product with the weight matrix
        act = arrayfun(@bipolar_activation, out); % apply the bipolar activation function
        disp('Input:')
        disp(arrayfun(@vis, reshape(S(:,i), 9, 7, 1)));
        disp('Target:');
        disp(arrayfun(@vis, reshape(T(:,i), 9, 7, 1)));
        disp('Output:');
        disp(arrayfun(@vis, reshape(act, 9, 7, 1)));
        disp(isequal(reshape(act, 9, 7, 1), reshape(T(:,i), 9, 7, 1))); % checks whether the output exactly matches the target
        i = i + 1;
    end

    % part c:
    disp('part c:')

    % change 12 random pixels in images 1, 3, 5, to 0's
    i = 1;
    while i <= 5
        sample = S(:,i);
        sample = add_noise(sample, 12);
        S(:,i) = sample;
        i = i + 2;
    end

    % testing loop
    i = 1;
    while i < length(S(1,:)) + 1
        out = S(:,i)' * W'; % pass each input through the network be computing the outer product with the weight matrix
        act = arrayfun(@bipolar_activation, out); % apply the bipolar activation function
        disp('Input:')
        disp(arrayfun(@vis, reshape(S(:,i), 9, 7, 1)));
        disp('Target:');
        disp(arrayfun(@vis, reshape(T(:,i), 9, 7, 1)));
        disp('Output:');
        disp(arrayfun(@vis, reshape(act, 9, 7, 1)));
        disp(isequal(reshape(act, 9, 7, 1), reshape(T(:,i), 9, 7, 1))); % checks whether the output exactly matches the target
        i = i + 1;
    end
end

% functions used in both Question 1 and Question 2:

function y = bipolar_activation(x) % bipolar activation function
    if x >= 0
        y = 1;
    else
        y = -1;
    end
end

% this function makes visuallizing the input/output images easier
function y = vis(x) % bipolar activation function, but 1 corresponds to #, and -1 to _
    if x > 0
        y = '#';
    elseif x == 0
        y = '*';
    else
        y = '_';
    end
end

% turns n random pixels in the input image to 0's
function r = add_noise(v, n)
    rand_indices = randperm(63, n); % generate n random indices from between 1 and 63
    for j = 1:n
        index = rand_indices(j);
        v(index) = 0;
    end
    r = v;
end

% mp
