# EE 456 HW2, Ian Deal, imd5205@psu.edu

# imports:
from scipy.io import loadmat # used for reading the .mat data files
from random import random, uniform, randint
import matplotlib.pyplot as plt # used for plotting the decision boundary
import pandas as pd # used for plotting the decision boundary
import numpy as np # used for plotting the decision boundary
from sklearn.metrics import accuracy_score # used for computing the final accuracy of the model

# flags for which problem to run:
problem1 = 0
problem2 = 0
problem3 = 1

# general functions used across networks:

# function that computes (w^T)*(x), where w and x are vectors
def wtx(w,x):
    out = 0
    for i in range(len(w)):
        out += w[i]*x[i]
    return out

# activation function for the perceptron network
def perceptron_act(v,theta):
    if v > theta:
        return 1
    elif v <= theta and v >= -1*theta:
        return 0
    else:
        return -1

# activation function for the MR2 networks
def mr2_act(v):
    if v >= 0:
        return 1
    else:
        return -1

# function that computes the weights of a single neuron using the delta rule
def delta_rule_update(alpha,weights,t,y_in,x):
    d_lst = [] # vector of deltas to return
    for n in range(len(weights)):
        delta = alpha*(t - y_in)*x[n]
        weights[n] += delta
        d_lst += [abs(delta)]
    return weights, d_lst

# Problem 1: Perceptron-based learning net.
if problem1 == 1:
    # load in the data
    data1 = loadmat('Two_moons_no_overlap.mat')
    data2 = loadmat('Two_moons_overlap.mat')
    train_x1, train_y1 = list(data1['X']), list(data1['Y'])
    train_x2, train_y2 = list(data2['X']), list(data2['Y'])
    train_x = train_x1 + train_x2
    train_y = train_y1 + train_y2

    # init weight vector
    weights = [uniform(0, 0) for _ in range(3)] # starting weights of 0
    bias = 0
    alpha = random() # random alpha between 0 and 1
    error = 1
    i = 0 # dataset iteration variable
    theta = 0.1 # activation threshold value
    total_error=[1]
    
    # begin training while there is an error
    while error > 0 and i < len(train_x):
        # compute the output ofthe network
        input = [bias]+list(train_x[i])
        v = wtx(weights, input)
        output = perceptron_act(v, theta) 
        error = abs(train_y[i][0] - output)

        # update the weights and bias following the perceptron algorithim
        for n in range(len(weights)):
            weights[n] += alpha*train_y[i][0]*input[n]
        bias += alpha * train_y[i][0]
        i += 2
        total_error +=[error]

    # plotting results and decision boundary
    # pass each point through the network and record the classifciations
    out_y=[]
    for i in range(len(train_y)):
        input = [bias]+list(train_x[i])
        v = wtx(weights, input)
        output = perceptron_act(v, theta) 
        out_y+=[output]

    # creates a dataframe of the x,y pairs and the model's classification
    data = pd.DataFrame({'x1': [coord[0] for coord in train_x], 'x2': [coord[1] for coord in train_x], 'class': out_y})
    
    # plot each pair on the scatterplot
    class_1 = data[data['class'] == 1]
    class_minus_1 = data[data['class'] == -1]
    plt.figure(figsize=(8, 6))
    plt.scatter(class_1['x1'], class_1['x2'], label='Class 1', c='blue', marker='o')
    plt.scatter(class_minus_1['x1'], class_minus_1['x2'], label='Class -1', c='red', marker='o')

    # compute boundary decision line
    m= (-weights[1])/weights[2]
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # define two points on the line based on the plot limits
    point1 = (x_min, m * x_min)
    point2 = (x_max, m * x_max)

    # plot the extended decision boundary line
    plt.fill_between([point1[0], point2[0]], [point1[1], point2[1]], y_min, color='red', alpha=0.2, label='Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    plt.plot(total_error)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()
    print('accuracy:', accuracy_score(out_y,train_y))


# Problem 2: MADALINE structure using MR2, 2 inputs, 1 hidden layer, and 1 output.
if problem2 == 1:
    # load in the data
    data1 = loadmat('Two_moons_no_overlap.mat')
    data2 = loadmat('Two_moons_overlap2.mat')
    train_x1, train_y1 = list(data1['X']), list(data1['Y'])
    train_x2, train_y2 = list(data2['X']), list(data2['Y'])
    train_x = train_x1 + train_x2
    train_y = train_y1 + train_y2

    # init weight vector, format= w_input_hidden[zI]: [bI,x1I,x2I] for I=[1:n]
    w_input_hidden = {'z1':[uniform(-0.05, 0.05) for _ in range(3)], 'z2':[uniform(-0.05, 0.05) for _ in range(3)], 'z3':[uniform(-0.05, 0.05) for _ in range(3)]} # weights connecting input to hidden neurons
    w_hidden_output =[-0.11, 0.1, 0.1, 0.1] # weights connecting hidden neurons to the output, forms a majority gate as > 2 neurons must fire for v to be >= 0
    hidden_neurons = {'z1':1, 'z2':2, 'z3':3} # lists the hidden neurons in the network and what index they occupy in the hiddden layer output vector
    bias = 1 # the unchanging bias value in the model, the weights for the biases are included in the weight vectors at idx = 0
    alpha = 0.005 
    delta_v = [1] 
    total_error = []
    i = 0 # dataset iteration variable
    c = 0 # count of training iterations
    # begin training network while max change in weights is > 0 and below the max-iteration threshold
    while max(delta_v) > 0.000009 and c<5555:
        hidden_output = [bias]
        hidden_in = {} # used to compute which hidden neuron's weights to update
        input = [bias]+list(train_x[i])
        for neuron in hidden_neurons:
            v = wtx(w_input_hidden[neuron], input)
            output = mr2_act(v)
            hidden_output += [output]
            while abs(v) in hidden_in: # ensures that there are no 'overlapping' values
                v+=0.0000001
            hidden_in[abs(v)] = [neuron,v]
        
        # compute the output of the network
        v = wtx(w_hidden_output, hidden_output)
        output = mr2_act(v)
        t = train_y[i][0]
        error = abs(t - output)
        
        # following step 7 of the MR2 algorithim
        if t != output and (-0.25 <= v and v <= 0.25):
            error_new = error + 1
            while error_new >= error and len(hidden_in)>0:
                abs_Z_in = min(hidden_in) # find the hidden neuron, Z, with the Z_in closest to 0
                Z = hidden_in[abs_Z_in][0]
                Z_in = hidden_in[abs_Z_in][1] 
                del hidden_in[abs_Z_in] # remove Z from the hidden_in dictionary so that it is not iterated over again
                Z_idx = hidden_neurons[Z] # obtain the index of Z's output within the hidden layer's output

                if hidden_output[Z_idx] >= 0:
                    hidden_output[Z_idx] = -1
                else:
                    hidden_output[Z_idx] = 1

                v_new = wtx(w_hidden_output, hidden_output) # recompute output with updated Z
                output_new = mr2_act(v_new)
                error_new = abs(t - output_new)
                # update weights of Z if the new error < previous error
                if error_new < error:
                    w = w_input_hidden[Z]
                    weights, delta = delta_rule_update(alpha,w,hidden_output[Z_idx],Z_in,input)
                    w_input_hidden[Z] = weights
                    delta_v = delta
                    break # ensures that only this neuron's weight is updated

        elif t != output:
            #update the weights using the delta rule
            for abs_Z_in in hidden_in:
                Z = hidden_in[abs_Z_in][0] # neuron to update
                Z_in = hidden_in[abs_Z_in][1] # the sum of the inputs into Z
                w = w_input_hidden[Z] # weight vector from the input to Z
                weights, delta = delta_rule_update(alpha,w,t,Z_in,input)
                w_input_hidden[Z] = weights
                delta_v = delta
        
        if i < 1000: # updates iteration variable to alternate between classes
            i+=1000
        else:
            i-=990
        total_error += [max(delta_v)]
        c+=1

    # plotting results and decision boundary
    # pass each point through the network and record the classifications
    out_y=[]
    for i in range(len(train_y)):
        hidden_output = [bias]
        input = [bias]+list(train_x[i])
        for neuron in hidden_neurons:
            v = wtx(w_input_hidden[neuron], input)
            output = mr2_act(v)
            hidden_output += [output]
        
        v = wtx(w_hidden_output, hidden_output)
        output = mr2_act(v)
        out_y += [output]

    # creates a dataframe of the x,y pairs and the model's classification
    data = pd.DataFrame({'x1': [coord[0] for coord in train_x], 'x2': [coord[1] for coord in train_x], 'class': out_y})
    
    # plot each pair on the scatterplot
    class_1 = data[data['class'] == 1]
    class_minus_1 = data[data['class'] == -1]
    plt.figure(figsize=(8, 6))
    plt.scatter(class_1['x1'], class_1['x2'], label='Class 1', c='blue', marker='o')
    plt.scatter(class_minus_1['x1'], class_minus_1['x2'], label='Class -1', c='red', marker='o')

    # compute the boundary decision lines and plot them
    m1 = (-w_input_hidden['z1'][1])/w_input_hidden['z1'][2]
    m2 = (-w_input_hidden['z2'][1])/w_input_hidden['z2'][2]
    m3 = (-w_input_hidden['z3'][1])/w_input_hidden['z3'][2]

    X = np.arange(-15, 25)
    plt.plot(X,m1*X,color='black',label='Decision Boundary')
    plt.plot(X,m2*X,color='black')
    plt.plot(X,m3*X,color='black')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    # plot error over time figure
    plt.plot(total_error)
    plt.xlabel('Iterations')
    plt.ylabel('Delta W')
    plt.show()
    print('accuracy:', accuracy_score(out_y,train_y)) # compute the accuracy of the model's classifications

# Problem 3: MADALINE structure using MR2, 2 inputs, 1 hidden layer, and 2 outputs.
if problem3 == 1:
    # load in the data
    data1 = loadmat('Two_moons_no_overlap2.mat')
    data2 = loadmat('Two_moons_overlap3.mat')
    train_x1, train_y1 = list(data1['X']), list(data1['Y'])
    train_x2, train_y2 = list(data2['X']), list(data2['Y'])
    train_x = train_x2
    train_y = train_y2

    # init weight vector, format= w_input_hidden[zI]: [bI,x1I,x2I] for I=[1:n]
    w_input_hidden = {'z1':[uniform(-0.05, 0.05) for _ in range(3)],'z2':[uniform(-0.05, 0.05) for _ in range(3)],'z3':[uniform(-0.05, 0.05) for _ in range(3)]} # weights connecting input to hidden neurons
    w_hidden_output ={'y1':[0.5]*4,'y2':[0.5]*4} # weights connecting hidden neurons to the output, forms an or gate
    hidden_neurons = {'z1':1,'z2':2,'z3':3} # lists the hidden neurons in the network and what index they occupy in the hiddden layer output vector
    output_neurons = ['y1','y2']
    bias = 1 # the unchanging bias value in the model, the weights for the biases are included in the weight vectors at idx = 0
    alpha = 0.05
    delta_v = [1]
    total_error = []
    i = 0 # dataset iteration variable
    c = 0 # count of training iterations
    # begin training network while max change in weights is > 0 and below the max-iteration threshold
    while max(delta_v) > 0.000009 and c<4555:
        hidden_output = [bias]
        hidden_in = {} # used to compute which hidden neuron's weights to update
        input = [bias]+list(train_x[i])
        for neuron in hidden_neurons:
            v = wtx(w_input_hidden[neuron], input)
            output = mr2_act(v)
            hidden_output += [output]
            while abs(v) in hidden_in: # ensures that there are no 'overlapping' values
                v+=0.00001
            hidden_in[abs(v)] = [neuron,v]
        
        # compute the output of the network
        output = []
        for neuron in output_neurons:
            v = wtx(w_hidden_output[neuron], hidden_output)
            out = mr2_act(v)
            output+=[out]
        
        t = train_y[i][0]
        error = abs(t - alpha*output[0]) + abs(t-alpha*output[1])

        # following step 7 of the MR2 algorithim
        if t != output[0] and t != output[1] and (-0.25 <= v and v <= 0.25):
            error_new = error + 1
            while error_new >= error and len(hidden_in)>0:
                abs_Z_in = min(hidden_in) # find the hidden neuron, Z, with the Z_in closest to 0
                Z = hidden_in[abs_Z_in][0]
                Z_in = hidden_in[abs_Z_in][1]
                del hidden_in[abs_Z_in] # remove Z from the hidden_in dictionary so that it is not iterated over again
                Z_idx = hidden_neurons[Z] # obtain the index of Z's output within the hidden layer's output

                if hidden_output[Z_idx] > 0: 
                    hidden_output[Z_idx] = -1
                else:
                    hidden_output[Z_idx] = 1

                # recompute output with updated Z
                output_new = []
                for neuron in output_neurons:
                    v = wtx(w_hidden_output[neuron], hidden_output)
                    out = mr2_act(v)
                    output_new +=[out]
                error_new = abs(t - output_new[0]) + abs(t - output_new[1])
                # update weights of Z if the new error < previous error
                if error_new < error:
                    w = w_input_hidden[Z]
                    weights, delta = delta_rule_update(alpha,w,hidden_output[Z_idx],Z_in,input)
                    w_input_hidden[Z] = weights
                    delta_v = delta
                    break # ensures that only this neuron's weight is updated

        elif t != output[0] and t != output[1]:
            #update the weights using the delta rule
            for abs_Z_in in hidden_in:
                Z = hidden_in[abs_Z_in][0] # neuron to update
                Z_in = hidden_in[abs_Z_in][1] # the sum of the inputs into Z
                w = w_input_hidden[Z] # weight vector from the input to Z
                weights, delta = delta_rule_update(alpha,w,t,Z_in,input)
                w_input_hidden[Z] = weights
            delta_v = delta
        total_error += [max(delta_v)]
        if i < len(train_x)-1: # updates iteration variable to alternate between classes
            i+=1
        else:
            i=0
        c+=1

    # plotting results and decision boundary
    # pass each point through the network and record the classifciations
    out_y=[]
    for i in range(len(train_y)):
        hidden_output = [bias]
        hidden_in = {} # used to compute which hidden neuron's weights to update
        input = [bias]+list(train_x[i])
        for neuron in hidden_neurons:
            v = wtx(w_input_hidden[neuron], input)
            output = mr2_act(v)
            hidden_output += [output]
            while abs(v) in hidden_in:
                v+=0.001
            hidden_in[abs(v)] = neuron
        
        output = []
        for neuron in output_neurons:
            v = wtx(w_hidden_output[neuron], hidden_output)
            out = mr2_act(v)
            output+=[out]
        if output[0] == output[1]:
            if output[0] == 1:
                out_y += [1]
            else:
                out_y += [-1]
        else:
            out_y += [0]

    # creates a dataframe of the x,y pairs and the model's classification
    data = pd.DataFrame({'x1': [coord[0] for coord in train_x], 'x2': [coord[1] for coord in train_x], 'class': out_y})
    
    # plot each pair on the scatterplot
    class_1 = data[data['class'] == 1]
    class_minus_1 = data[data['class'] == -1]
    plt.figure(figsize=(8, 6))
    plt.scatter(class_1['x1'], class_1['x2'], label='Class 1', c='blue', marker='o')
    plt.scatter(class_minus_1['x1'], class_minus_1['x2'], label='Class 2', c='green', marker='o')

    # compute the boundary decision lines and plot them
    m1 = (-w_input_hidden['z1'][1])/w_input_hidden['z1'][2]
    m2 = (-w_input_hidden['z2'][1])/w_input_hidden['z2'][2]
    m3 = (-w_input_hidden['z3'][1])/w_input_hidden['z3'][2]
    X = np.arange(-14, 24)
    plt.plot(X,m1*X,color='black',label='Decision Boundary')
    plt.plot(X,m2*X,color='black')
    plt.plot(X,m3*X,color='black')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

    # plot error over time figure
    plt.plot(total_error)
    plt.xlabel('Iterations')
    plt.ylabel('Delta W')
    plt.show()
    print('accuracy:', accuracy_score(out_y,train_y)) # compute the accuracy of the model's classifications