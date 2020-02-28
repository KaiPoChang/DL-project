#!/usr/bin/env python
# coding: utf-8

# In[9]:


# coding: utf-8

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
from torch.autograd import Variable
import re
import random
import os
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Create dictionary & Hyperparameter

# In[10]:


# set a dictionary (char2num) for alphabet and SOS, EOS token
all_letters = string.ascii_lowercase
letters = dict()
n=0
for i in all_letters:
    letters[i] = n
    n+=1
letters["SOS_token"]= 26
letters["EOS_token"]= 27
print(letters)

#----------Hyper Parameters----------#
hidden_size = 256
#The number of vocabulary
vocab_size = 28
teacher_forcing_ratio = 0.5
KLD_weight = 0

max_length=28
learning_rate=0.0001
n_classes=4
latent_dim = 256
output_dim = vocab_size


# # Helper function

# In[11]:


def wordTotensor(word):
    words = list(word)
    lengthOfword = len(words)
    tensor = torch.zeros(lengthOfword)
    for i in range(lengthOfword):
        tensor[i] = letters[words[i]]
    return tensor

#compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    return sentence_bleu([reference], output, smoothing_function=cc.method1)

def calculate_loss(x, reconstructed_x, mean, log_var):
    
    # reconstruction loss
    criterion = nn.CrossEntropyLoss()
    
    # RCL = nn.CrossEntropyLoss(reconstructed_x, x)
    RCL=criterion(reconstructed_x, x)
    
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) # a=1
    return RCL , KLD


# # Encoder

# In[12]:


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, latent_dim, n_classes):

        super().__init__()

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size + n_classes)
        self.hidden_size = hidden_size
        
        self.mu = nn.Linear(hidden_size + n_classes, latent_dim)
        self.var = nn.Linear(hidden_size + n_classes, latent_dim)

    def forward(self, input_tensor, hidden):
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        a=output
        d=hidden
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# # Decoder

# In[13]:


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, latent_dim, n_classes):
        super().__init__()      
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size + n_classes)
        self.out = nn.Linear(hidden_size + n_classes, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor , hidden):
        output = self.embedding(input_tensor).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# # CVAE

# In[14]:


class CVAE(nn.Module):
    
    def __init__(self, input_size, hidden_dim, latent_dim, n_classes, output_size):

        super().__init__()

        self.encoder = Encoder(input_size, hidden_size, latent_dim, n_classes)
        self.decoder = Decoder(hidden_size, output_size, latent_dim, n_classes)
        self.mu = nn.Linear(hidden_size + n_classes, latent_dim)
        self.var = nn.Linear(hidden_size + n_classes, latent_dim)

    def forward(self, input_word, condition1, condition2, Wtargrt):
        
        lw=0
        if epoch>0:
            KLD_weight=0.01+0.01*lw

        lw+=1

        sp=np.array([1, 0, 0, 0])
        tp=np.array([0, 1, 0, 0]) 
        pg=np.array([0, 0, 1, 0]) 
        p=np.array([0, 0, 0, 1])

        ## Encode ##
        
        # convert the word to index by letters
        input_tensor = wordTotensor(input_word)
        input_tensor = Variable(input_tensor)
        input_length = input_tensor.size(0)
        target_tensor = wordTotensor(Wtargrt)
        target_length = wordTotensor(Wtargrt).size(0)
        
        sp=sp[np.newaxis, np.newaxis] # [[[1, 0, 0, 0]]] 
        tp=tp[np.newaxis, np.newaxis] # [[[0, 1, 0, 0]]]
        pg=pg[np.newaxis, np.newaxis] # [[[0, 0, 1, 0]]]
        p=p[np.newaxis, np.newaxis] # [[[0, 0, 0, 1]]]
        
        #initial the hidden_layer
        init_hidden = self.encoder.initHidden() 
        init_hidden = torch.cuda.FloatTensor(init_hidden)
        init_hidden = Variable(init_hidden)
        
        # concatenate init_hidden with conditional_tense
        if condition1==0:
            init_hidden = torch.cat((init_hidden, torch.from_numpy(sp).float().cuda()), dim=-1)
        elif condition1==1:
            init_hidden = torch.cat((init_hidden, torch.from_numpy(tp).float().cuda()), dim=-1)
        elif condition1==2:
            init_hidden = torch.cat((init_hidden, torch.from_numpy(pg).float().cuda()), dim=-1)
        else:
            init_hidden = torch.cat((init_hidden, torch.from_numpy(p).float().cuda()), dim=-1)

        # split index_word to index_char    
        for i in range(input_length):
            input_tensors = input_tensor[i]
            input_tensors = input_tensors.detach().numpy()
            input_tensors = torch.cuda.LongTensor(input_tensors)
            
            # put concatenated_hidden with index_char into encoder
            encoder_output, encoder_hidden = self.encoder(input_tensors, init_hidden)
        
        z_mu = self.mu(encoder_hidden)
        z_var = self.var(encoder_hidden)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterization trick
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        
        # concatenate the conditional tense with x_sample => conditional_x_sample
        if condition2==0:
            conditional_x_sample = torch.cat((x_sample, torch.from_numpy(sp).float().cuda()), dim=-1)
        elif condition2==1:
            conditional_x_sample = torch.cat((x_sample, torch.from_numpy(tp).float().cuda()), dim=-1)
        elif condition2==2:
            conditional_x_sample = torch.cat((x_sample, torch.from_numpy(pg).float().cuda()), dim=-1)
        else:
            conditional_x_sample = torch.cat((x_sample, torch.from_numpy(p).float().cuda()), dim=-1)
            
        decoder_input = torch.cuda.LongTensor([[letters["SOS_token"]]])
        
        ## Decoder ##
        
        generated_x=[]
        
        hidden_z = conditional_x_sample
        hidden_z = hidden_z.cpu().detach().numpy()
        hidden_z = torch.cuda.FloatTensor(hidden_z)
        
        loss=0
        # Teacher forcing: Feed the target as the next input
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            for di in range(target_length):
                # put SOS (decoder_input) and hidden_z into decoder
                decoder_output, hidden_z = self.decoder(decoder_input, hidden_z)
                # calculate loss
                RCL, KLD = calculate_loss(torch.cuda.LongTensor(target_tensor[di].unsqueeze(0).long().cuda()), 
                                          decoder_output, z_mu, z_var)
                
                loss += (KLD_weight*KLD+RCL)
                decoder_o = torch.max(decoder_output, 1) 
                decoder_out = decoder_o[1]
                # generated_x: collect the maxium probability of char (list)
                generated_x.append(decoder_out.cpu().detach().numpy())
                # Teacher forcing
                decoder_input =torch.cuda.LongTensor(target_tensor[di].unsqueeze(0).long().cuda())  
                
        # Without teacher forcing: use its own predictions as the next input    
        else:
            for di in range(target_length):
                decoder_output, hidden_z = self.decoder(decoder_input, hidden_z)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                RCL, KLD = calculate_loss(torch.cuda.LongTensor(target_tensor[di].unsqueeze(0).long().cuda(0)), 
                                       decoder_output, z_mu, z_var)
               
                loss += (KLD_weight*KLD + RCL)
                # optimizer.step()
                

                decoder_o=torch.max(decoder_output, 1) 
                decoder_input=decoder_o[1]
                generated_x.append(decoder_input.cpu().detach().numpy())
                # generated_x.append(decoder_input.cpu().detach().numpy())
                if decoder_input.item() == letters["EOS_token"]:
                    break

        return generated_x, z_mu, z_var, KLD, loss


# # Generate data

# In[15]:


def generateData(lang1):
    print("Reading %s data..."% lang1)
    lines = open('%s.txt' % lang1, encoding='utf-8').read().strip().split('\n')
    data = [l for l in lines]
    return data

# generate training data
pairs=generateData('train')
Training_pairs = [l.split(" ") for l in pairs]
print(Training_pairs)

# generate testing data
pairs=generateData('test')
Testing_pairs = [l.split(" ") for l in pairs]
print(Testing_pairs)


# # Start to train

# In[16]:


# model
model = CVAE(input_size=28, hidden_dim=256, latent_dim = 256, n_classes=4, output_size=28)
# model.load_state_dict(torch.load("29modict.pth"))
model.cuda()


#optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[17]:


# making reversed letters dictionary (num2char)
lookup = dict()
n=0
for i in all_letters:
    lookup[n] = i
    n+=1
lookup[26]= "SOS_token"
lookup[27]= "EOS_token"
print(lookup)


# In[18]:


epoch=80

KLDLoss=[]
TotalLoss=[]
BLEU4_scores_training=[]
TrainingScoresEpoch=[]
Scores=[]
ScoresEpoch=[]

for i in range(epoch):
    model.train()
    print("Epoch",i)
    
    for u in range(1200):
        for j in range(4):
            optimizer.zero_grad()
            
            condition1=j
            condition2=j

            generated_x, z_mu, z_var, KLD, loss = model.forward(Training_pairs[u][j], 
                                                                condition1, condition2, Training_pairs[u][j])
            loss.backward()
            optimizer.step()
            
            xx = ""
            indx = 0
            for y in generated_x:
                if indx < len(generated_x)-1:
                    xx = xx + lookup[y[0]]
                indx += 1
            
            BLEU4_score_training = compute_bleu(xx, Training_pairs[u][j])
            BLEU4_scores_training.append(BLEU4_score_training)

            
    KLDLoss.append(KLD.cpu().detach().numpy())
    TotalLoss.append(loss.cpu().detach().numpy())

    model.eval()
    for q in range(len(Testing_pairs)):
        # set the testing conditional tense in advance 
        condition1 = np.array([0,0,0,0,3,0,3,2,2,2])
        condition2 = np.array([3,2,1,1,1,2,0,0,3,1])
        generated_x, z_mu, z_var, KLD, loss = model.forward(Testing_pairs[q][0],
                                                            condition1[q], condition2[q], Testing_pairs[q][1])
        # collect prediction
        prediction = ""
        ind = 0
        for v in generated_x:
            if ind < len(generated_x)-1:
                prediction = prediction + lookup[v[0]]

            ind+=1

        print("Input:", Testing_pairs[q][0], "； Prediction:", prediction, "； True:", Testing_pairs[q][1])
        Score = compute_bleu(prediction, Testing_pairs[q][1])
        Scores.append(Score)
  
    
    TrainingScore=np.sum(BLEU4_scores_training)/len(BLEU4_scores_training)
    TrainingScoresEpoch.append(TrainingScore)
    
    Scores10=np.sum(Scores)/len(Scores)
    ScoresEpoch.append(Scores10)
    
    # saving the models
    path1 = "".join(str(i)+'model.pth')
    path2 = "".join(str(i)+'modict.pth')
    
    torch.save(model, path1)
    torch.save(model.state_dict(), path2)
    
# saving lists for plotting
np.save("KLDLoss.npy", KLDLoss)
np.save("ScoresEpoch.npy", ScoresEpoch)
np.save("TotalLoss.npy", TotalLoss)
np.save("TrainingScore.npy", TrainingScoresEpoch)


# # Plot figure

# In[44]:


plt.plot(TrainingScoresEpoch)
plt.title("BLEU4 Training scores of each Epoch")
plt.xlabel("epoch")
plt.savefig("BLEU4_score_Training")
plt.show()
print('Best BLEU4 TrainingScore: ', np.max(TrainingScoresEpoch))

plt.plot(ScoresEpoch)
plt.title("BLEU4 Testing Scores of each Epoch")
plt.xlabel("epoch")
plt.savefig("BLEU4_score_Testing")
plt.show()
print('Best BLEU4 Testing Scores: ', np.max(ScoresEpoch))

plt.plot(KLDLoss)
plt.title("KLD Loss")
plt.xlabel("epoch")
plt.savefig("KLDLoss")
plt.show()

plt.plot(TotalLoss)
plt.title("Total Loss")
plt.xlabel("epoch")
plt.savefig("TotalLoss")
plt.show()


# # Word Generation

# In[37]:


decoder_input = torch.cuda.LongTensor([[letters["SOS_token"]]])
generated_x = []
for i in range(5):
    hidden_z = torch.rand(1, 1, 256).cuda() # create random 1*1*260 array 
    hidden_z = torch.cat((hidden_z, torch.from_numpy(np.array([[[0, 1, 0, 0]]])).float().cuda()), dim=-1)
    hidden_z = torch.cuda.FloatTensor(hidden_z)
    DECODER = Decoder(hidden_size=256, output_size=28, latent_dim=256, n_classes=4)
    decoder_output, hidden_z = DECODER(decoder_input.cpu(), hidden_z.cpu())
    decoder_o = torch.max(decoder_output, 1) 
    decoder_out = decoder_o[1]
    # generated_x: collect the maxium probability of char (list)
    generated_x.append(decoder_out.cpu().detach().numpy())
    
ind = 0
prediction = ''
for i in generated_x:
    if ind < len(generated_x)-1:
        prediction = prediction + lookup[i[0]]
    ind+=1
print(prediction)


# In[ ]:




