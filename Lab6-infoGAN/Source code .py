#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import torchvision.utils as vutils

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image


# ## Hyperparameter

# In[2]:


batch_size = 100
D_lr = 2e-4
Q_lr = 1e-3
G_lr = 1e-3
nz = 64
nc = 10 # (size of meaningful codes)
ngf = 64
ndf = 64
Total_epochs = 80
# optimizer =  Adam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Model

# In[3]:


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# ## FrontEnd

# In[4]:


class FrontEnd(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self):
    super(FrontEnd, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1, 64, 4, 2, 1),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(128, 256, 4, 2, 1, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(256, 512, 4, 2, 1, bias=False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2, inplace=True)
    )

  def forward(self, x):
    output = self.main(x) 
    return output


# In[5]:


FrontEnd()


# ## Discriminator

# In[6]:


class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()

        self.main = nn.Sequential(
          nn.Conv2d(512, 1, 4, 1, bias=False),
          nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x).view(-1, 1)
        return output


# ## Generator

# In[7]:


class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()

        self.main = nn.Sequential(
          nn.ConvTranspose2d(64, 512, 4, 1, bias=False),
          nn.BatchNorm2d(512),
          nn.ReLU(True),
          nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
          nn.BatchNorm2d(256),
          nn.ReLU(True),
          nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(True),
          nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(True),
          nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
          nn.Tanh()
        )

    def forward(self, x):
        output = self.main(x)
        return output


# In[8]:


G()


# ## Classifier

# In[9]:


class Q(nn.Module):

    def __init__(self):
        super(Q, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features = 8192, out_features = 100, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = 100, out_features = 10, bias = True)
        )

    def forward(self, x):
        x = x.view(-1, 8192)
        output = self.main(x)
        return output


# In[10]:


FE = FrontEnd()
D = D()
Q = Q()
G = G()
for i in [FE, D, Q, G]:
    i.cuda()
    i.apply(weights_init)


# # Data preprocessing 

# In[11]:


dataset = dset.MNIST(root='./dataset', download=False,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


# # Train

# In[12]:


criterionD = nn.BCELoss().cuda()
criterionQ_dis = nn.CrossEntropyLoss().cuda()

real_x = torch.FloatTensor(batch_size, 1, 80, 80).cuda()
label = torch.FloatTensor(batch_size, 1).cuda()
dis_c = torch.FloatTensor(batch_size, 10).cuda()
con_c = torch.FloatTensor(batch_size, 2).cuda()
noise = torch.FloatTensor(batch_size, 62).cuda()

real_x = Variable(real_x)
label = Variable(label, requires_grad=False)
dis_c = Variable(dis_c)
con_c = Variable(con_c)
noise = Variable(noise)

# fixed random variables
c = np.linspace(-1, 1, 10).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)

idx = np.arange(10).repeat(10)
one_hot = np.zeros((100, 10))
one_hot[range(100), idx] = 1 
fix_noise = torch.Tensor(100, 54).uniform_(-1, 1) 

# setup optimizer
optimD = optim.Adam([{'params':FE.parameters()}, {'params': D.parameters()}], lr=0.00002, betas=(0.5, 0.99))
optimG = optim.Adam([{'params':G.parameters()}, {'params': Q.parameters()}], lr=0.001, betas=(0.5, 0.99))


# In[13]:


def _noise_sample(dis_c, noise, bs):
    idx = np.random.randint(10, size=bs)# len(idx) = 100
    c = np.zeros((bs, 10)) # c.shape = (100, 10)
    c[range(bs),idx] = 1.0

    dis_c.data.copy_(torch.Tensor(c)) # (100, 10)
    noise.data.uniform_(-1.0, 1.0) # (100, 54)
    z = torch.cat([noise, dis_c], 1).view(100, 64, 1, 1)
    
    return z, idx


# In[14]:


G_LOSS = []
D_LOSS = []


# In[15]:


for epoch in range(80):
    for num_iters, batch_data in enumerate(dataloader, 0):
        # real part
        optimD.zero_grad()

        x, _ = batch_data # x.size = ([100, 1, 64, 64])

        bs = x.size(0)
        real_x.data.resize_(x.size())
        label.data.resize_(bs, 1)
        dis_c.data.resize_(bs, 10)
        noise.data.resize_(bs, 54)

        real_x.data.copy_(x) # real_x.size = ([100, 1, 64, 64])
        fe_out1 = FE(real_x) # fe_out1.size = ([100, 512, 4, 4])
        probs_real = D(fe_out1) # probs_real.size = ([100, 1])
        label.data.fill_(1)
        loss_real = criterionD(probs_real, label)
        loss_real.backward()

        # fake part
        z, idx = _noise_sample(dis_c, noise, bs) # z.size=([100, 64, 1, 1])； len(idx)=100
        fake_x = G(z) # fake_x.size = ([64, 1, 64, 64])
        fe_out2 = FE(fake_x.detach()) # fe_out2.size = ([100, 512, 4, 4])
        probs_fake = D(fe_out2) # probs_fake.size([100, 1]) 
        label.data.fill_(0)
        loss_fake = criterionD(probs_fake, label)
        loss_fake.backward()

        D_loss = loss_real + loss_fake
        D_LOSS.append(D_loss)
        
        optimD.step()

        ## D and Q part ##
        optimG.zero_grad()
        
        # adversarial loss
        fe_out = FE(fake_x) # fe_out.size = ([100, 512, 4, 4])
        probs_fake = D(fe_out) # probs_fake.size = ([100, 1])
        label.data.fill_(1.0)

        reconstruction_loss = criterionD(probs_fake, label)
        
        # mutual info
        q_output = Q(fe_out) # q_output.size = ([100, 10])
        class_ = torch.LongTensor(idx).cuda() # class_.size = ([100])
        target = Variable(class_)
        dis_loss = criterionQ_dis(q_output, target)

        G_loss = reconstruction_loss + dis_loss 
        G_LOSS.append(G_loss)
        
        G_loss.backward()
        optimG.step()

        if num_iters % 100 == 0:

            print('Epoch/Iter: {0}/{1}, Dloss: {2}, Gloss: {3}'.format(
            epoch, num_iters, D_loss.data.cpu().numpy(),
            G_loss.data.cpu().numpy())
            )

            noise.data.copy_(fix_noise) # fix_noise.size = ([64, 54])
            dis_c.data.copy_(torch.Tensor(one_hot))

            z = torch.cat([noise, dis_c], 1).view(100, 64, 1, 1)
            x_save = G(z)
            save_image(x_save.data, '%s/real_samples.png' % '.', normalize=True, nrow=10)
    
    # do checkpointing
    torch.save(G.state_dict(), '%s/model/G_epoch_%d.pth' % ('.', epoch))
    torch.save(D.state_dict(), '%s/model/D_epoch_%d.pth' % ('.', epoch))


# In[16]:


print(len(G_LOSS))
print(len(D_LOSS))


# In[53]:


import matplotlib.pyplot as plt
epoch = np.arange(0, 48000)

plt.figure('D_LOSS', figsize=(25, 5))
plt.plot(epoch, D_LOSS, label='D_LOSS', color='blue')

plt.title('D LOSS')
plt.xlabel('Epoch')
plt.ylabel('LOSS')

my_x_ticks = np.arange(0, 48000, 2000)
plt.xticks(my_x_ticks)

plt.legend()
plt.savefig('D_LOSS.png')
plt.show() 


# In[51]:


epoch = np.arange(0, 48000)

plt.figure('G_LOSS', figsize=(25, 5))
plt.plot(epoch, G_LOSS, label='Q_LOSS', color='red')

plt.title('G LOSS')
plt.xlabel('Epoch')
plt.ylabel('LOSS')

my_x_ticks = np.arange(0, 48000, 2000)
plt.xticks(my_x_ticks)

plt.legend()
plt.savefig('G_LOSS.png')
plt.show() 

