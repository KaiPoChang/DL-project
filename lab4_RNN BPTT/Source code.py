#!/usr/bin/env python
# coding: utf-8

# # Build RNN Model

# In[142]:


## initialize parameters
class RNNNumpy():
    def __init__(self, word_dim, word_batch_dim, hidden_dim=16, bptt_truncate=4):
        # assign instance variable
        self.word_dim = word_dim
        self.word_batch_dim = word_batch_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # random initiate the parameters
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_batch_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        
    ## 1. forward propagation
    def softmax(self, x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)

    def forward_propagation(self, x):
        # total num of time steps, len of vector x
        T = len(x) # T = 2
        # during forward propagation, save all hidden stages in s, S_t = U .dot x_t + W .dot s_{t-1}
        # we also need the initial state of s, which is set to 0
        # each time step is saved in one row in s，each row in s is s[t] which corresponding to an rnn internal loop time
        s = np.zeros((T, self.hidden_dim)) # s.shape = (2, 16)
        s[-1] = np.zeros(self.hidden_dim)
        # output at each time step saved as o, save them for later use
        o = np.zeros((T, self.word_dim)) # o.shape = (2, 16)
        for t in np.arange(T):
            # we are indexing U by x[t]. it is the same as multiplying U with a one-hot vector
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = self.softmax(self.V.dot(s[t]))
#             print('s[t]', s[t].shape)
#             print('o[t]', o[t].shape)
        return [o, s]
    
    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis = 1)
    
    def calculate_total_loss(self, x, y):
        L = 0
        # for each sentence ...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i]) # x[i] = (1, 2)
            # we only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(y[i]), y[i]]
            # add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
            # print(L)
        return L

    def calculate_loss(self, x, y):
        # divide the total loss by the number of training examples
        N = len(y) # N = 8
        return self.calculate_total_loss(x, y)/N

    ## 3. BPTT
    def bptt(self, x, y):
        T = 1
        # perform forward propagation
        o, s = self.forward_propagation(x)
        # we will accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(T), y] -= 1   # it is y_hat - y
        # for each output backwards ...
        for t in np.arange(T):
            dLdV += np.outer(delta_o[t], s[t].T)    # at time step t, shape is word_dim * hidden_dim
            # initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t]**2))
            # backpropagation through time (for at most self.bptt_truncate steps)
            # given time step t, go back from time step t, to t-1, t-2, ...
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # update delta for next step
                dleta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1]**2)
        return [dLdU, dLdV, dLdW]
    
            
    def numpy_sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW


# ## Generate binary data

# In[143]:


import numpy as np
import random
from datetime import datetime
from random import randint
random.seed(10)

x1 = ''.join([str(randint(0, 1)) for i in range(0, 8)])
x2 = ''.join([str(randint(0, 1)) for i in range(0, 8)])
print(x1, x2)

x1_train = np.array([int(i) for i in x1])
x2_train = np.array([int(i) for i in x2])

x_train = np.concatenate((np.transpose(x1_train), np.transpose(x2_train)), axis=0).reshape(8, 2)
y_train = list(bin(int(x1, 2) + int(x2, 2)))[-8:]
y_train = np.array([int(i) for i in y_train]).reshape(8,)
print(x_train, y_train)
print(x_train.shape, y_train.shape)


# In[144]:


def train_with_sgd(model, X_train, y_train, learning_rate = 0.005, nepoch = 100, evaluate_loss_after = 5):
    # keep track of the losses so that we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: loss after num_examples_seen=%d epoch=%d: %f" %(time, num_examples_seen, epoch, loss))
        # for each training example...
        for i in range(len(y_train)):
            # one sgd step
            model.numpy_sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
        
    return losses


# In[151]:


word_dim = 16
hidden_dim = 16
word_batch_dim = 2

np.random.seed(0)

model = RNNNumpy(word_dim, word_batch_dim, hidden_dim)

losses = train_with_sgd(model, x_train, y_train, nepoch = 13000, evaluate_loss_after = 5)


# In[152]:


import matplotlib.pyplot as plt
epoch = np.arange(0, 13000, 5)
accuracy = [(100-loss) for num_examples, loss in losses]

plt.figure()
plt.plot(epoch, accuracy, label='Training Accuracy', color='blue')

plt.title('Training Accuracy (BPTT)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.legend()
plt.show() 

