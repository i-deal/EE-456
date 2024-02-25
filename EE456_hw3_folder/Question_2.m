% EE 456 HW3, Ian Deal, imd5205@psu.edu

clear all; % Clear all variables
clc; % Clear the command window
close all; % Close all figure windows

question_2 = 1;

% run the second question
if question_2 == 1
    disp('question 2:')
    % Part a:
    disp('part a:')

    load("Data\S_patterns_HW3_P2.mat","S");
    n = 4; % number of sample images
    s1 = S(:,:,1); % image of a 1
    s2 = S(:,:,10); % image of a 0
    s3 = S(:,:,7); % image of a 7
    s4 = S(:,:,5); % image of a 5

    S = [s1, s2, s3, s4];
    S = reshape(S, 63, n); % flattens the S matrix for easier forward pass computing
    W = zeros(63, 63); % init weight matrix
    i = 1;

    % training weights using the Hebb rule
    while i < length(S(1,:)) + 1 % iterate through all sample images
        x = S(:,i); % the ith sample image
        j = 1;
        % Hebb rule, update the weight vector W_j by adding the product of x_j and x
        while j < 64
            delta = x(j) * x; % x_j times the target of x
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
        disp('Output:');
        disp(arrayfun(@vis, reshape(act, 9, 7, 1)));
        disp(isequal(reshape(act, 9, 7, 1), reshape(S(:,i), 9, 7, 1))); % checks whether the output exactly matches the target
        i = i + 1;
    end

    % Part b:
    disp('part b:')

    load("Data\S_patterns_HW3_P2.mat","S");
    n = 4; % number of sample images
    s1 = S(:,:,1); % image of a 1
    s2 = S(:,:,10); % image of a 0
    s3 = S(:,:,7); % image of a 7
    s4 = S(:,:,5); % image of a 5

    S = [s1, s2, s3, s4];
    S = reshape(S, 63, n); % flattens the S matrix for easier forward pass computing
    W = zeros(63, 63); % init weight matrix

    % change pixels
    i = 1;
    while i <= n
        sample = S(:,i);
        sample = add_noise(sample, 1);
        S(:,i) = sample;
        i = i + 1;
    end

    % training weights using the Hebb rule
    i = 1;
    while i < length(S(1,:)) + 1 % iterate through all sample images
        x = S(:,i); % the ith sample image
        j = 1;
        % Hebb rule, update the weight vector W_j by adding the product of x_j and x
        while j < 64
            delta = x(j) * x; % x_j times the target of x
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
        disp('Output:');
        disp(arrayfun(@vis, reshape(act, 9, 7, 1)));
        i = i + 1;
   end

    % Part c:
    disp('part c:')

    load("Data\S_patterns_HW3_P2.mat","S");
    n = 4; % number of sample images
    s1 = S(:,:,6); % image of a 6
    s2 = S(:,:,10); % image of a 0
    s3 = S(:,:,8); % image of a 8
    s4 = S(:,:,5); % image of a 5

    S = [s1, s2, s3, s4];
    S = reshape(S, 63, n); % flattens the S matrix for easier forward pass computing
    W = zeros(63, 63); % init weight matrix

    % initial training of weights using the Hebb rule
    i = 1;
    while i < length(S(1,:)) + 1 % iterate through all sample images
        x = S(:,i); % the ith sample image
        j = 1;
        % Hebb rule, update the weight vector W_j by adding the product of x_j and x
        while j < 64
            delta = x(j) * x; % x_j times the target of x
            W(:,j) = W(:,j) + delta; % update the weight vector, W_j
            j = j + 1;
        end
        i = i + 1;
    end
    
    % reiterating through each sample image and re-training instances of imperfect recall
    i = 1;
    while i < length(S(1,:)) + 1 % iterate through all sample images
        out = S(:,i)' * W';
        act = arrayfun(@bipolar_activation, out);
        if isequal(reshape(act, 9, 7, 1), reshape(S(:,i), 9, 7, 1)) == 0
            x = S(:,i); % the ith sample image
            j = 1;
            % Hebb rule, update the weight vector W_j by adding the product of x_j and x
            while j < 64
                delta = x(j) * x; % x_j times the target of x
                W(:,j) = W(:,j) + delta; % update the weight vector, W_j
                j = j + 1;
            end
        end
        out = S(:,i)' * W'; % pass each input through the network be computing the outer product with the weight matrix
        act = arrayfun(@bipolar_activation, out); % apply the bipolar activation function
        if isequal(reshape(act, 9, 7, 1), reshape(S(:,i), 9, 7, 1))
            i = i + 1;
        end
    end
    
    % testing loop
    i = 1;
    while i < length(S(1,:)) + 1
        out = S(:,i)' * W'; % pass each input through the network be computing the outer product with the weight matrix
        act = arrayfun(@bipolar_activation, out); % apply the bipolar activation function
        disp('Input:')
        disp(arrayfun(@vis, reshape(S(:,i), 9, 7, 1)));
        disp('Output:');
        disp(arrayfun(@vis, reshape(act, 9, 7, 1)));
        disp(isequal(reshape(act, 9, 7, 1), reshape(S(:,i), 9, 7, 1))); % checks whether the output exactly matches the target
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