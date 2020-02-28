
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid


import math
import random

from PIL import Image, ImageOps, ImageEnhance
import numbers

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label


# In[3]:


train_data, train_label, test_data, test_label = read_bci_data()


# In[4]:


train_data = torch.from_numpy(train_data)
train_label = torch.from_numpy(train_label)
test_data = torch.from_numpy(test_data)
test_label = torch.from_numpy(test_label)


# In[5]:


class EEGNet(nn.Module):
    def __init__(self, activation):
        super(EEGNet, self).__init__()
        activations = nn.ModuleDict([['LeakyReLU', nn.LeakyReLU()], 
                                     ['ReLU', nn.ReLU()],
                                     ['ELU', nn.ELU()]
                                    ])

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activations[activation],
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activations[activation],
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
    def forward(self, x):
        out = self.firstconv(x)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out = out.view(out.size(0), -1) # flatten the output of conv2 to (batch_size, 736)
        out = self.classify(out)
        return out


# In[6]:


class DeepConvNet(nn.Module):
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()
        activations = nn.ModuleDict([['LeakyReLU', nn.LeakyReLU()], 
                                     ['ReLU', nn.ReLU()],
                                     ['ELU', nn.ELU()]
                                    ])
        
        self.firstlayer = nn.Sequential(# (1, 2, 750)
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 0), bias=False), # (25, 2, 746)
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), padding=(0, 0), bias=False), # (25, 1, 746)
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activations[activation],
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),  # (25, 1, 373)
            nn.Dropout(p=0.5)
        )
        self.secondlayer = nn.Sequential(# (25, 1, 373)
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), bias=False), # (50, 1, 369)
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activations[activation],
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1), padding=0), # (50, 1, 368)
            nn.Dropout(p=0.5)
        )
        self.thirdlayer = nn.Sequential( # (50, 1, 368)
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), bias=False), # (100, 1, 364)
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            activations[activation],
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1), padding=0), # (100, 1, 363)
            nn.Dropout(p=0.5)
        )
        self.forthlayer = nn.Sequential( # (100, 1, 363)
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), bias=False), # (200, 1, 359)
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activations[activation],
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1), padding=0), # (200, 1, 358)
            nn.Dropout(p=0.5)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=200*1*358, out_features=2, bias=True)
        )
    def forward(self, x):
        out = self.firstlayer(x)
        out = self.secondlayer(out)
        out = self.thirdlayer(out)
        out = self.forthlayer(out) # (batch, 200, 1, 358)
        out = out.view(out.size(0), -1) # flatten the output of conv2 to (batch_size, 200*1*358)
        out = self.classify(out)
        return out


# In[7]:


EEGNet_ELU = EEGNet(activation='ELU')
EEGNet_ELU = EEGNet_ELU.float()
print(EEGNet_ELU)
print('============================================================================================================')
DeepConvNet_ELU = DeepConvNet(activation='ELU')
DeepConvNet_ELU = DeepConvNet_ELU.float()
print(DeepConvNet_ELU)


# In[8]:


EEGNet_LeakyReLU = EEGNet(activation='LeakyReLU')
EEGNet_LeakyReLU = EEGNet_LeakyReLU.float()
print(EEGNet_LeakyReLU)
print('============================================================================================================')
DeepConvNet_LeakyReLU = DeepConvNet(activation='LeakyReLU')
DeepConvNet_LeakyReLU = DeepConvNet_LeakyReLU.float()
print(DeepConvNet_LeakyReLU)


# In[9]:


EEGNet_ReLU = EEGNet(activation='ReLU')
EEGNet_ReLU = EEGNet_ReLU.float()
print(EEGNet_ReLU)
print('============================================================================================================')
DeepConvNet_ReLU = DeepConvNet(activation='ReLU')
DeepConvNet_ReLU = DeepConvNet_ReLU.float()
print(DeepConvNet_ReLU)


# In[10]:


# Hyper Parameters
batch_size = 16
learning_rate = 1e-3
num_epochs = 400
loss_func = nn.CrossEntropyLoss()


# In[11]:


def predicted(NN):
    
    # optimize all cnn parameters
    optimizer = torch.optim.Adam(NN.parameters(), lr=learning_rate)
    
    Accuracy_train = []
    Accuracy_test = []
    for epoch in range(num_epochs):
        train_data_batch = Variable(train_data).float()
        train_label_batch = Variable(train_label).float()
        #獲取最後輸出
        train_out = NN(train_data_batch)
        #獲取loss
        train_loss = loss_func(train_out, train_label_batch.long())
        #使用optimizer optimize loss (清空上一步殘餘更新參數)
        optimizer.zero_grad()
        #Backpropagation，計算參數更新值
        train_loss.backward()
        #將參數更新後的值加到NN的parameters上
        optimizer.step()
            
        # test
        if epoch % 2 == 0:
            test_data_batch = Variable(test_data).float()
            test_label_batch = Variable(test_label)
        
            test_out = NN(test_data_batch)
            accuracy_test = torch.max(test_out,1)[1].numpy() == test_label_batch.numpy() #accuracy=True/False 
            Accuracy_test.append(accuracy_test.mean()) 
            
            accuracy_train = torch.max(train_out,1)[1].numpy() == train_label_batch.numpy() 
            Accuracy_train.append(accuracy_train.mean())
            
            print('{} round: accuracy_train ='.format(epoch), accuracy_train.mean())
            print('{} round: accuracy_test ='.format(epoch), accuracy_test.mean())
    return Accuracy_train, Accuracy_test


# In[12]:


print('(1)EEGNet_ELU_result:')
EEGNet_ELU_train, EEGNet_ELU_test = predicted(EEGNet_ELU)
print('\n(2)EEGNet_ReLU_result:')
EEGNet_ReLU_train, EEGNet_ReLU_test = predicted(EEGNet_ReLU)
print('\n(3)EEGNet_LeakyReLU_result:')
EEGNet_LeakyReLU_train, EEGNet_LeakyReLU_test = predicted(EEGNet_LeakyReLU)


# In[13]:


epoch = np.arange(0, num_epochs, 2)

plt.figure('EEGNet_ELU_Accuracy')
plt.plot(epoch, EEGNet_ELU_test, label='Accuracy_ELU_test', color='blue')
plt.plot(epoch, EEGNet_ELU_train, label='Accuracy_ReLU_train', color='red')
plt.plot(epoch, EEGNet_ReLU_test, label='Accuracy_LeakyReLU_test', color='green')
plt.plot(epoch, EEGNet_ReLU_train, label='Accuracy_ELU_train', color='magenta')
plt.plot(epoch, EEGNet_LeakyReLU_test, label='Accuracy_ReLU_test', color='yellow')
plt.plot(epoch, EEGNet_LeakyReLU_train, label='Accuracy_LeakyReLU_train', color='black')

plt.title('Activation function comparison(EEGNet)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.show() 


# In[14]:


print('(4)DeepConvNet_ELU_result:')
DeepConvNet_ELU_train, DeepConvNet_ELU_test = predicted(DeepConvNet_ELU)
print('(5)DeepConvNet_ReLU_result:')
DeepConvNet_ReLU_train, DeepConvNet_ReLU_test = predicted(DeepConvNet_ReLU)
print('(6)DeepConvNet_LeakyReLU_result:')
DeepConvNet_LeakyReLU_train, DeepConvNet_LeakyReLU_test = predicted(DeepConvNet_LeakyReLU)


# In[15]:


epoch = np.arange(0, num_epochs, 2)

plt.figure('EEGNet_ELU_Accuracy')
plt.plot(epoch, DeepConvNet_ELU_test, label='Accuracy_ELU_test', color='blue')
plt.plot(epoch, DeepConvNet_ELU_train, label='Accuracy_ReLU_train', color='red')
plt.plot(epoch, DeepConvNet_ReLU_test, label='Accuracy_LeakyReLU_test', color='green')
plt.plot(epoch, DeepConvNet_ReLU_train, label='Accuracy_ELU_train', color='magenta')
plt.plot(epoch, DeepConvNet_LeakyReLU_test, label='Accuracy_ReLU_test', color='yellow')
plt.plot(epoch, DeepConvNet_LeakyReLU_train, label='Accuracy_LeakyReLU_train', color='black')

plt.title('Activation function comparison(DeepConvNet)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.show() 


# In[16]:


print('(1) EEGNet_ELU_MAX_accuracy_test_result:', max(EEGNet_ELU_test))
print('(2) EEGNet_ReLU_MAX_accuracy_test_result:', max(EEGNet_ReLU_test))
print('(3) EEGNet_LeakyReLU_MAX_accuracy_test_result:', max(EEGNet_LeakyReLU_test))
print('=============================================================================')
print('(4) DeepConvNet_ELU_MAX_accuracy_test_result:', max(DeepConvNet_ELU_test))
print('(5) DeepConvNet_ReLU_MAX_accuracy_test_result:', max(DeepConvNet_ReLU_test))
print('(6) DeepConvNet_LeakyReLU_MAX_accuracy_test_result:', max(DeepConvNet_LeakyReLU_test))

