clear all; % Clear all variables
clc; % Clear the command window
close all; % Close all figure windows
x = 0;

if x == 1
    load("Data\SandT_patterns_HW3_P1.mat", "S");
    
    S = reshape(S, 63, 3); % flattens the S matri
    s4 = [-1 1 1 1 1 1 -1; -1 1 -1 -1 -1 1 -1; -1 1 -1 -1 -1 1 -1; 1 -1 1 -1 1 -1 1; -1 1 -1 -1 -1 1 -1; 1 -1 1 -1 1 -1 1; -1 1 -1 -1 -1 1 -1; -1 1 -1 -1 -1 1 -1; -1 1 1 1 1 1 -1;];
    s4 = reshape(s4, 63, 1);
    
    o1 = dot(S(:,1),S(:,2)) + dot(S(:,1),S(:,3));
    o2 = dot(S(:,2),S(:,1)) + dot(S(:,2),S(:,3));
    o3 = dot(S(:,3),S(:,1)) + dot(S(:,3),S(:,2));
    o4 = dot(s4,S(:,1)) + dot(s4,S(:,2)) + dot(s4,S(:,3));
    
    heatmap(reshape(s4, 9, 7, 1));
    disp(o1/2);
    disp(o2/2);
    disp(o3/2);
    disp(o4/3);
else
    load('sampleweight.mat','W');
    W1=normalize(W(:,1));
    W2=normalize(W(:,2));
    disp(W1);
    disp(W2);
    load('samplematrix.mat','S');
    i=3;
    d1 = norm(S(:,i)-W1)^2;
    d2 = norm(S(:,i)-W2')^2;
    disp(d1);
    disp(d2);
    disp((0.6*S(:,i))+W1);
    x = normalize((0.6*S(:,i))+W1);
    disp(x)
    %W(:,1) = x;
    %save('sampleweight1.mat','W');
end

if x == 3
    load('samplematrix.mat','S');
    load('sampleweight.mat','W');
    d1 = norm(S(:,1)-W(:,1))^2;
    w2 = W(:,2)';
    d2 = norm(S(:,1)-w2')^2;
    disp(d1);
    disp(d2);
    disp(0.6*(S(:,1)-w2')+w2');
end