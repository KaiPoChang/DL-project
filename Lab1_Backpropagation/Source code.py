
# coding: utf-8

# In[55]:


def generate_linear(n=100):
    import numpy as np
    import random
    #Draw samples from a uniform distribution.；np.random.uniform(low=0.0, high=1.0, size=None)
    random.seed(1)
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


# In[56]:


def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
            
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(21,1)


# In[57]:


from math import exp
from random import seed
from random import random


# In[58]:


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


# In[59]:


# Forward propagate input to a network output
def forward_propagate(network, row):
    '''
    依照每一個row(dataset)為input，按照layer的neuron以weight相加為activation，輸入到transfer裡用neuron['output']存起來，
    再append到new_inputs裡面，所以new_inputs就代表是當層layer的output，也就是下一層layer的input，一個layer會有一組的new_inputs，
    return的inputs為outputlayer的output
    '''
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# In[60]:


def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


# In[61]:


def derivative_sigmoid(output):
    return output * (1.0 - output)


# In[62]:


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    '''
    一開始的迴圈會由後往前做layer的運算，當此層layer為outputlayer的時候，i == len(network)-1，所以走else，運算出來的值會存在errors裡面，
    可以乘上transfer_derivative(neuron['output'])，這個值存在neuron裡面，以neuron['delta']存起來，再來下一層layer走if，
    對該層layer的neuron而言，它的error為，計算前面一層layer的weight和delta (error)，再把它append到errors。

    '''
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * derivative_sigmoid(neuron['output'])
    


# In[63]:


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# In[64]:


# Train a network for a fixed number of epochs
def train_network(network, dataset, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        outputs_copy = []
        for row in dataset:
            outputs = forward_propagate(network, row)
            outputs_copy = outputs.copy()
            #outputlayer有2個neuron，若dataset中ground truth為1的標註為[1,0]，0的標註[0, 1]
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            sum_error += sum([(expected[i]-outputs[i]) for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        if epoch%1000 == 0:
            print(' epoch '+ str(epoch) + ' loss : '+ str(sum_error))
            print(outputs_copy)


# In[65]:


def initialize_network(n_inputs, n_hidden1, n_hidden2, n_outputs):
    network = list()
    hidden_layer1 = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden1)]
    network.append(hidden_layer1)
    hidden_layer2 = [{'weights':[random() for i in range(n_hidden1 + 1)]} for i in range(n_hidden2)]
    network.append(hidden_layer2)
    output_layer = [{'weights':[random() for i in range(n_hidden1 + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# In[66]:


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# In[67]:


def back_propagation(dataset, n_inputs, n_hidden1, n_hidden2, n_outputs, l_rate, n_epoch):
    network = initialize_network(n_inputs, n_hidden1, n_hidden2, n_outputs)
    train_network(network, dataset, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in dataset:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)


# In[68]:


def show_result(x, y, pred_y):
    import matplotlib.pyplot as plt
    
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]): #x.shape[0] 算出x維度
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.show()


# In[69]:


import numpy as np
x1, y1 = generate_linear(n=100)
dataset1 = np.concatenate((x1, y1), axis=1).tolist()
x2, y2 = generate_XOR_easy()
dataset2 = np.concatenate((x2, y2), axis=1).tolist()


# In[70]:


predictions = back_propagation(dataset1, 2, 4, 4, 2, 0.3, 20000)
pred_y1 = list(predictions)
show_result(x1, y1, pred_y1)


# In[71]:


Round=0
count_truth=0
ground_truth = [int(raw[-1])for raw in dataset1]
for i in predictions:
    if i == ground_truth[Round]:
        count_truth += 1
    Round += 1
print(count_truth/len(predictions))


# In[76]:


predictions = back_propagation(dataset2, 2, 4, 4, 2, 0.3, 20000)
pred_y2 = list(predictions)
show_result(x2, y2, pred_y2)


# In[77]:


Round=0
count_truth=0
ground_truth = [int(raw[-1])for raw in dataset2]
for i in predictions:
    if i == ground_truth[Round]:
        count_truth += 1
    Round += 1
print(count_truth/len(predictions))

